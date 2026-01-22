from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the RAG components
from hybrid_graph_rag import chain, rag_chart

# FastAPI app initialization
app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body format
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []  # [["user msg", "ai msg"], ...]

# ---- RAG API Endpoint ----
@app.post("/api/query")
async def query_rag(request: QueryRequest):
    question = request.question
    chat_history = request.chat_history

    try:
        # Note: Chart requests should go to /api/chart endpoint
        # This endpoint handles regular text queries
        
        # ---- Normal RAG Answer ----
        result = chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })

        return {"answer": result}

    except Exception as e:
        print("Backend Error:", e)
        return {"answer": f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question."}

# ---- Root endpoint ----
@app.get("/")
async def home():
    return {"status": "Hybrid Graph RAG backend running successfully!"}

from fastapi.responses import Response
import io
import base64
from hybrid_graph_rag import rag_chart_image

@app.post("/api/chart")
async def chart_api(request: QueryRequest):
    """
    Generate a chart based on the user's question.
    Returns chart image as base64 or error message.
    """
    try:
        img_bytes = rag_chart_image(request.question)

        # base64 encode for sending to frontend
        encoded = base64.b64encode(img_bytes).decode("utf-8")

        return {
            "image": encoded,
            "answer": "Chart generated successfully"
        }

    except ValueError as ve:
        # User-friendly error message for missing data
        error_message = str(ve)
        return {
            "answer": error_message,
            "error": error_message,
            "image": None
        }
    except Exception as e:
        error_message = f"Unable to generate chart: {str(e)}"
        print(f"Chart generation error: {e}")
        return {
            "answer": error_message,
            "error": error_message,
            "image": None
        }


# ---- Start server ----
if __name__ == "__main__":
    print(" Starting Backend API at http://localhost:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
