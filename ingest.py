# Library for reading and extracting text from PDF files
from pypdf import PdfReader
# Qdrant vector database client
from qdrant_client import QdrantClient
# Qdrant models for configuring vector storage
from qdrant_client.models import VectorParams, Distance, PointStruct
# Configuration constants for Qdrant connection and collection name
from config import QDRANT_URL, COLLECTION_NAME
# Utility function to generate embeddings from text
from utils import embed_text
# UUID library for generating unique identifiers
import uuid


# Initialize Qdrant client to connect to the vector database
client = QdrantClient(url=QDRANT_URL)


# Create a vector collection in Qdrant if it doesn't already exist
# Parameters:
#   vector_size: the dimensionality of the embeddings (e.g., 384 for embedding model)
def create_collection(vector_size: int):
    # Check if collection already exists to avoid duplicates
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        # Create new collection with cosine distance similarity metric
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


# Extract all text content from a PDF file
# Parameters:
#   path: file path to the PDF document
# Returns:
#   text: concatenated text from all pages
def read_pdf(path: str):
    # Initialize PDF reader with the given file path
    reader = PdfReader(path)
    text = ""
    # Iterate through all pages in the PDF and extract text
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# Split text into overlapping chunks for processing by the embedding model
# Parameters:
#   text: the full text to be chunked
#   chunk_size: number of characters per chunk (default: 500)
# Returns:
#   chunks: list of text chunks
def chunk_text(text, chunk_size=500):
    chunks = []
    # Create chunks by iterating through the text with fixed stride
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


# Main function to ingest a PDF document into the vector database
# This process consists of:
#   1. Extract text from PDF
#   2. Split text into chunks
#   3. Generate embeddings for each chunk
#   4. Store embeddings and text in Qdrant
# Parameters:
#   path: file path to the PDF document to be ingested
def ingest_pdf(path: str):
    # Step 1: Read and extract all text from the PDF
    text = read_pdf(path)
    # Step 2: Split text into manageable chunks
    chunks = chunk_text(text)

    # Step 3: Generate embedding for the first chunk to determine embedding dimensions
    first_embedding = embed_text(chunks[0])
    # Create vector collection with appropriate dimensionality
    create_collection(len(first_embedding))

    # Initialize list to store vector points for bulk insertion
    points = []

    # Step 4: Generate embeddings for all chunks and prepare data for storage
    for chunk in chunks:
        # Convert text chunk to vector embedding
        embedding = embed_text(chunk)

        # Create a point structure with unique ID, vector, and original text
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": chunk}
            )
        )

    # Upload all points (embeddings + text) to the Qdrant vector database
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    # Print confirmation message
    print("Ingestão concluída.")
