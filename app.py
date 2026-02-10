# FastAPI framework for building REST APIs
from fastapi import FastAPI, UploadFile, File
# Module for high-level file operations
import shutil
# Function to ingest and process PDF documents
from ingest import ingest_pdf
# Function to build the RAG (Retrieval-Augmented Generation) workflow graph
from rag_graph import build_graph

# Initialize FastAPI application instance
app = FastAPI()
# Build the LangGraph workflow for RAG pipeline
graph = build_graph()


# Endpoint to handle PDF file uploads and indexing
# POST /upload - receives a PDF file and processes it for RAG
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Create a temporary file path with the uploaded filename
    path = f"temp_{file.filename}"

    # Write the uploaded file to disk
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the PDF: extract text, create embeddings, and index in vector database
    ingest_pdf(path)

    # Return success response
    return {"status": "Documento indexado com sucesso"}


# Endpoint to handle question answering against indexed documents
# POST /ask - receives a question and returns an answer based on retrieved context
@app.post("/ask")
async def ask_question(question: str):
    # Execute the RAG graph workflow with the user's question
    # Input includes the question, empty context (will be populated by retrieval step), 
    # and empty answer (will be filled by generation step)
    result = graph.invoke({
        "question": question,
        "context": [],
        "answer": ""
    })

    # Extract and return the generated answer from the workflow result
    return {"answer": result["answer"]}
