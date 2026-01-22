from dotenv import load_dotenv
import os
from langchain_neo4j import Neo4jGraph

from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

from langchain_neo4j import Neo4jVector

from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars

from langchain_google_genai import ChatGoogleGenerativeAI 

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()



chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0  
)

# AURA_INSTANCENAME = os.environ["AURA_INSTANCENAME"] # No longer needed for local setup
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Only needed if using OpenAI

kg = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# create vector index
vector_index = Neo4jVector.from_existing_graph(
    hf,
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)

class Entities(BaseModel):
    """Entity information extracted from text."""
    names: List[str] = Field(
        description="List of person, organization, or business entities appearing in the text"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system",
 "Extract entities including persons, organizations, business units, financial terms (CapEx, revenue, EBITDA), \
 metrics, time periods, and economic indicators from the text. Return only meaningful entities."),
    ("human", "Extract all the entities from the following input: {question}")
])

entity_chain = prompt | chat.with_structured_output(Entities)

# Test entity extraction
entities = entity_chain.invoke({"question": "How has the company's capital expenditure (CapEx) evolved over the past five years, and what strategic initiatives have driven significant changes in CapEx allocations"})
print(entities.names)

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    words = [w for w in remove_lucene_chars(input).split() if w]
    if not words:
        return ""
    return " AND ".join(f"{word}~2" for word in words)

def structured_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        print(f" Getting Entity: {entity}")
        # Using standard pattern matching instead of fulltext search
        response = kg.query(
            """
            MATCH (node)
            WHERE node.name =~ $query 
            OR node.id =~ $query
            WITH node
            MATCH (node)-[r]->(neighbor)
            RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
            UNION ALL
            MATCH (node)<-[r]-(neighbor)
            WHERE node.name =~ $query 
            OR node.id =~ $query
            RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            LIMIT 50
            """,
            {"query": f"(?i).*{entity}.*"}  # Case-insensitive pattern matching
        )
        result += "\n".join([el["output"] for el in response])
    return result
# Final retrieval step
def retriever(question: str):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question)
    unstructured_data = [
        el.page_content for el in vector_index.similarity_search(question)
    ]
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    print(f"\nFinal Data::: ==>{final_data}")
    return final_data

import re
from collections import Counter
import matplotlib.pyplot as plt

# --- RAG Chain Condenser (Handles Chat History) ---
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

# Branch to decide whether to condense the question or use it as-is
_search_query = RunnableBranch(
    # Check if chat history exists
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),
        # Condense question with history
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | chat
        | StrOutputParser(),
    ),
    # Else, pass through the original question
    RunnableLambda(lambda x: x["question"]),
)
# --- Final RAG Chain ---
template = """Answer the question based only on the following context:
{context}

Question: {question}
Use natural language and be concise.
Answer:"""
final_rag_prompt = ChatPromptTemplate.from_template(template)

# The main RAG execution chain
chain = (
    RunnableParallel(
        {
            # Note: _search_query handles rephrasing, then feeds to the retriever
            "context": _search_query | retriever,
            "question": RunnablePassthrough(),
        }
    )
    | final_rag_prompt
    | chat
    | StrOutputParser()
)
# --- Chart Generation Function ---
def rag_chart(question, chart_type="bar"):
    """
    Generates a chart based on the RAG chain's response to a question.
    """
    print("\n Asking RAG:", question)

    # Invoke the RAG chain to get the response
    response = chain.invoke({
        "question": question,
        "chat_history": []
    })

    print("\nRAG Response:\n", response)

    # Try extracting numbers first
    numbers = re.findall(r"(\d+\.\d+|\d+)", response)
    values = [float(n) for n in numbers]

    # If numeric data is insufficient, fallback to keyword frequency
    if not values or len(values) < 2:
        print("\n No numeric data found. Switching to keyword frequency mode.\n")

        # Extract keywords and count frequency
        words = re.findall(r"[A-Za-z]{4,}", response.lower())
        word_counts = Counter(words)
        keywords = ["risk", "competition", "innovation", "market", "supply", "security", "privacy", "sustainability"]

        # Filter counts to only include the defined keywords
        labels = [k for k in keywords if word_counts[k] > 0]
        values = [word_counts[k] for k in labels]

        if not labels:
            print("Could not generate a meaningful chart as no keywords were found.")
            return

    else:
        # Simple use case: Numbers found, use generic labels
        labels = [f"Value {i+1}" for i in range(len(values))]


    # Plotting
    plt.figure(figsize=(8,5))

    if chart_type == "bar":
        plt.bar(labels, values)
        plt.title(f"Bar Chart for: {question[:50]}...")
        plt.ylabel("Value / Frequency")
    elif chart_type == "pie":
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title(f"Pie Chart for: {question[:50]}...")
        plt.axis('equal')
    else:
        print("Invalid chart type specified. Use 'bar' or 'pie'.")
        return

    plt.show()
    print("\n Chart generated successfully!\n")

if __name__ == "__main__":
    print("--- Hybrid Graph RAG Runner Initialized ---")
    
    # --- Example Query 1 (Text Answer) ---
    question_1 = "What long-term investments and strategic initiatives does Apple highlight in its 2024 report?"
    print(f"\n[Running Query 1: {question_1}]")
    try:
        # Note: Chat history is empty for the first question
        result_1 = chain.invoke({"question": question_1, "chat_history": []})
        print(f"\nAnswer 1:\n{result_1}\n")
    except Exception as e:
        print(f"An error occurred during Query 1: {e}")

    # --- Example Query 2 (Chart Generation) ---
    question_2 = "Create a chart showing the number of risk factors across categories."
    print(f"\n[Running Query 2 (Chart): {question_2}]")
    try:
        # Note: rag_chart calls the RAG chain internally
        rag_chart(question_2, "bar")
    except Exception as e:
        print(f"An error occurred during Query 2: {e}")