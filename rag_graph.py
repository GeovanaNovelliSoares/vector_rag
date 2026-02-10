# LangGraph library for building state machine workflows
from langgraph.graph import StateGraph
# Type hints for function parameters and TypedDict definitions
from typing import TypedDict, List
# Qdrant vector database client for similarity search
from qdrant_client import QdrantClient
# Configuration constants for Qdrant, collection name, and LLM model
from config import QDRANT_URL, COLLECTION_NAME, LLM_MODEL
# Utility functions for text embedding and answer generation
from utils import embed_text, generate_answer


# Initialize Qdrant client to connect to the vector database
client = QdrantClient(url=QDRANT_URL)


# TypedDict defining the state structure for the RAG workflow
# This state is passed between nodes in the graph
class GraphState(TypedDict):
    # The user's question/query
    question: str
    # List of retrieved context chunks from the vector database
    context: List[str]
    # The final generated answer from the LLM
    answer: str


# Retrieval node: searches the vector database for relevant context
# This is the first step in the RAG pipeline
# Parameters:
#   state: current graph state containing the user's question
# Returns:
#   updated state with retrieved context chunks
def retrieve(state: GraphState):
    # Convert the user's question into a vector embedding
    embedding = embed_text(state["question"])

    # Query the Qdrant vector database for the 5 most similar document chunks
    # Using cosine similarity to measure relevance
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=5
    ).points

    # Extract the original text from the retrieved results
    context = [r.payload["text"] for r in results]

    # Return updated state with the retrieved context
    return {"context": context}



# Generation node: uses the LLM to generate an answer based on retrieved context
# This is the second step in the RAG pipeline
# Parameters:
#   state: current graph state containing the question and retrieved context
# Returns:
#   updated state with the generated answer
def generate(state: GraphState):
    # Combine all context chunks into a single text block separated by newlines
    context_text = "\n\n".join(state["context"])

    # Construct a prompt that guides the LLM to answer based only on the provided context
    prompt = f"""
    Você é um assistente especializado em análise técnica/jurídica.
    Responda apenas com base no contexto abaixo.

    CONTEXTO:
    {context_text}

    PERGUNTA:
    {state['question']}
    """

    # Call the LLM to generate an answer using the prompt and model from config
    answer = generate_answer(prompt, LLM_MODEL)

    # Return updated state with the generated answer
    return {"answer": answer}


# Build the RAG workflow graph using LangGraph
# Creates a state machine with two nodes: retrieve and generate
# Execution flow: retrieve -> generate
# Returns:
#   compiled graph ready for execution
def build_graph():
    # Initialize a state graph with the defined GraphState structure
    graph = StateGraph(GraphState)

    # Add the retrieval node to the graph
    graph.add_node("retrieve", retrieve)
    # Add the generation node to the graph
    graph.add_node("generate", generate)

    # Set the retrieval node as the starting point of the workflow
    graph.set_entry_point("retrieve")
    # Define the edge from retrieval to generation (workflow order)
    graph.add_edge("retrieve", "generate")

    # Compile the graph into an executable workflow
    return graph.compile()
