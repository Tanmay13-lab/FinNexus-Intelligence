# Complete Setup Guide for Friend - Using Neo4j Desktop

This guide will help you set up and run the Hybrid Graph RAG Financial Analyzer project on your system using **Neo4j Desktop (local database)**.

---

## 📋 Prerequisites

Before starting, make sure you have:
- ✅ Windows/Mac/Linux system
- ✅ Python 3.9 or 3.10 installed
- ✅ Node.js installed (version 14 or higher)
- ✅ Internet connection

---

## 🔧 STEP 1: Install Neo4j Desktop

1. **Download Neo4j Desktop:**
   - Go to: https://neo4j.com/download/
   - Click "Download Neo4j Desktop"
   - Download the installer for your operating system

2. **Install Neo4j Desktop:**
   - Run the downloaded installer
   - Follow the installation wizard
   - Create a Neo4j account (if you don't have one) - it's FREE

3. **Launch Neo4j Desktop:**
   - Open Neo4j Desktop application
   - Sign in with your Neo4j account

---

## 🗄️ STEP 2: Create Local Neo4j Database

1. **Create a New Project:**
   - In Neo4j Desktop, click "+ New Project"
   - Name it: "Financial Analyzer" (or any name you like)
   - Click "Create"

2. **Add a Local Database:**
   - Inside your project, click "+ Add" → "Local DBMS"
   - Database name: `financial-analyzer` (or any name)
   - Password: Choose a password (REMEMBER THIS - you'll need it!)
   - Version: Use the latest stable version (e.g., 5.x or 4.x)
   - Click "Create"

3. **Start the Database:**
   - Click on your newly created database
   - Click the "Start" button (play icon)
   - Wait until status changes to "Running" (green)

4. **Note the Connection Details:**
   - Once running, you'll see connection details like:
     - **Bolt URI:** `bolt://localhost:7687`
     - **Username:** `neo4j` (default)
     - **Password:** The password you set above
   - **KEEP THIS INFORMATION** - you'll need it for `.env` file

---

## 📁 STEP 3: Extract Project Files

1. **Extract the ZIP file** your friend sent you
2. **Navigate to the project folder:**
   - Open the extracted folder
   - Go inside: `Hybrid-Graph-RAG-Financial-Analyser-main`
   - You should see folders like: `backend_api`, `frontend`, `Annual Report`, etc.

---

## ⚙️ STEP 4: Create .env File

1. **In the project root folder**, create a file named `.env`

2. **Open `.env` file** in a text editor and add these lines:

   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_database_password_here
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

3. **Replace the values:**
   - `your_database_password_here` → The password you set in STEP 2.3
   - `your_gemini_api_key_here` → Your Google Gemini API key (see STEP 5)

4. **Save the `.env` file**

---

## 🔑 STEP 5: Get Google Gemini API Key

1. **Go to Google AI Studio:**
   - Visit: https://makersuite.google.com/app/apikey
   - Or: https://aistudio.google.com/app/apikey

2. **Sign in** with your Google account

3. **Create API Key:**
   - Click "Create API Key"
   - Choose "Create API key in new project" or select existing project
   - Copy the API key shown

4. **Add to .env file:**
   - Paste the API key in your `.env` file (replace `your_gemini_api_key_here`)

---

## 🐍 STEP 6: Install Python Dependencies

1. **Open Command Prompt / Terminal**

2. **Navigate to project folder:**
   ```bash
   cd "path\to\Hybrid-Graph-RAG-Financial-Analyser-main"
   ```

3. **Check if Python is installed:**
   ```bash
   python --version
   ```
   - If you see version number (e.g., "Python 3.10.5"), continue
   - If you see error "python is not recognized", see **TROUBLESHOOTING** section below

4. **Install required packages (try these in order):**

   **Option 1 - Using python -m pip (RECOMMENDED):**
   ```bash
   python -m pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-google-genai langchain-neo4j langchain-experimental pydantic matplotlib numpy faiss-cpu sentence-transformers
   ```

   **Option 2 - Using py launcher (Windows):**
   ```bash
   py -m pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-google-genai langchain-neo4j langchain-experimental pydantic matplotlib numpy faiss-cpu sentence-transformers
   ```

   **Option 3 - Direct pip (if above don't work):**
   ```bash
   pip install fastapi uvicorn python-dotenv langchain langchain-community langchain-google-genai langchain-neo4j langchain-experimental pydantic matplotlib numpy faiss-cpu sentence-transformers
   ```

5. **Wait for installation to complete** (may take 2-5 minutes)

---

## 📊 STEP 7: Import Graph Data into Neo4j

**IMPORTANT:** Your friend should have sent you a file called `graph_documents.pkl`. Make sure it's in the project root folder.

1. **Open Python in the project folder** (or use VS Code / Jupyter)

2. **Create a new Python file:** `import_data.py`

3. **Copy this code into `import_data.py`:**

   ```python
   from dotenv import load_dotenv
   import os
   import pickle
   from langchain_neo4j import Neo4jGraph
   from langchain_community.embeddings import HuggingFaceEmbeddings
   from langchain_neo4j import Neo4jVector

   # Load environment variables
   load_dotenv()

   # Connect to Neo4j
   kg = Neo4jGraph(
       url=os.environ["NEO4J_URI"],
       username=os.environ["NEO4J_USERNAME"],
       password=os.environ["NEO4J_PASSWORD"],
   )

   print("Connected to Neo4j successfully!")

   # Load graph documents from pickle file
   print("Loading graph documents...")
   with open("graph_documents.pkl", "rb") as f:
       graph_documents = pickle.load(f)

   print(f"Loaded {len(graph_documents)} graph documents.")

   # Import graph documents into Neo4j
   print("Importing graph documents into Neo4j...")
   kg.add_graph_documents(
       graph_documents,
       include_source=True,
       baseEntityLabel=True,
   )

   print("Graph documents imported successfully!")

   # Create vector index
   print("Creating vector index...")
   model_name = "sentence-transformers/all-mpnet-base-v2"
   model_kwargs = {'device': 'cpu'}
   encode_kwargs = {'normalize_embeddings': False}

   hf = HuggingFaceEmbeddings(
       model_name=model_name,
       model_kwargs=model_kwargs,
       encode_kwargs=encode_kwargs
   )

   vector_index = Neo4jVector.from_existing_graph(
       hf,
       search_type="hybrid",
       node_label="Document",
       text_node_properties=["text"],
       embedding_node_property="embedding",
   )

   print("Vector index created successfully!")
   print("\n✅ Setup complete! Your Neo4j database is ready.")
   ```

4. **Run the script:**
   ```bash
   python import_data.py
   ```

5. **Wait for completion** (this may take 5-10 minutes):
   - Loading graph documents
   - Importing into Neo4j
   - Creating vector index

6. **Verify import:**
   - Go to Neo4j Desktop
   - Click "Open" button on your database (opens Neo4j Browser)
   - Run query: `MATCH (n) RETURN count(n) as node_count`
   - You should see nodes imported (count should be > 0)

---

## 🚀 STEP 8: Run the Backend

1. **Open Command Prompt / Terminal**

2. **Navigate to backend folder:**
   ```bash
   cd backend_api
   ```

3. **Start the backend server:**
   ```bash
   python backend_api.py
   ```

4. **Wait for message:**
   - You should see: `Starting Backend API at http://localhost:5000`
   - **Keep this terminal open** - backend must keep running

---

## 🎨 STEP 9: Run the Frontend

1. **Open a NEW Command Prompt / Terminal** (keep backend running!)

2. **Navigate to frontend folder:**
   ```bash
   cd frontend
   ```

3. **Install frontend dependencies (first time only):**
   ```bash
   npm install
   ```
   Wait for completion (1-2 minutes)

4. **Start the frontend:**
   ```bash
   npm start
   ```

5. **Browser should open automatically:**
   - If not, manually go to: `http://localhost:3000`

---

## ✅ STEP 10: Test the Application

1. **You should see the chat interface**

2. **Try asking a question:**
   - Example: "What is Apple's revenue?"
   - Example: "Show me revenue by product category as a bar chart"

3. **If it works, you're done!** 🎉

---

## 🛑 How to Stop the Application

**To stop:**

1. **Stop Frontend:**
   - In frontend terminal, press `Ctrl + C`

2. **Stop Backend:**
   - In backend terminal, press `Ctrl + C`

3. **Stop Neo4j Database (optional):**
   - In Neo4j Desktop, click "Stop" button on your database

---

## ❗ Troubleshooting

### Problem: "Cannot connect to Neo4j"
**Solution:**
- Make sure Neo4j database is running (green "Running" status)
- Check `.env` file has correct `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Try changing `bolt://localhost:7687` to `neo4j://localhost:7687` in `.env`

### Problem: "Module not found" error
**Solution:**
- Run `pip install` again with all packages from STEP 6

### Problem: "graph_documents.pkl not found"
**Solution:**
- Ask your friend to send you the `graph_documents.pkl` file
- Place it in the project root folder (same folder as `import_data.py`)

### Problem: "npm command not found"
**Solution:**
- Install Node.js from https://nodejs.org/

### Problem: "Port 5000 or 3000 already in use"
**Solution:**
- Close other applications using these ports
- Or kill the process: `taskkill /F /IM python.exe` and `taskkill /F /IM node.exe`

---

## 📝 Summary

**To run the project, you need:**
1. ✅ Neo4j Desktop installed and database running
2. ✅ `.env` file with correct credentials
3. ✅ Graph data imported into Neo4j (STEP 7)
4. ✅ Backend running (Terminal 1)
5. ✅ Frontend running (Terminal 2)

**Every time you want to use the app:**
1. Start Neo4j Desktop → Start your database
2. Run backend: `cd backend_api` → `python backend_api.py`
3. Run frontend: `cd frontend` → `npm start`

---

**Need Help?** Contact your friend who sent you this project! 😊

