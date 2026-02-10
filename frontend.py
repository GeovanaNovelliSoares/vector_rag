# Streamlit UI library for building web apps
import streamlit as st
# HTTP client library for making API requests
import requests
# Time module for sleep and timing operations
import time

# Backend API endpoint for RAG service
API_URL = "http://localhost:8000"

# Configure the Streamlit page settings (title, icon, layout)
st.set_page_config(
    page_title="VectorMind - RAG",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS styling for dark theme and layout customization
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
.chat-container {
    max-width: 900px;
    margin: auto;
}
.status-card {
    padding: 10px;
    border-radius: 10px;
    background-color: #1c1f26;
    margin-bottom: 10px;
}
.small-text {
    font-size: 0.85rem;
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# Initialize chat message history in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []


# Create sidebar with system controls and service status
with st.sidebar:
    st.title("⚙️ Sistema")

    if st.button("🧹 Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔍 Serviços")

    # Function to check if a service is online by making a GET request
    # Returns a green dot if online, red dot if offline
    def check_service(url):
        try:
            requests.get(url, timeout=2)
            return "🟢 Online"
        except:
            return "🔴 Offline"

    # Display status of all required services
    st.markdown(f"Ollama: {check_service('http://localhost:11434')}")
    st.markdown(f"Backend: {check_service('http://localhost:8000/docs')}")
    st.markdown(f"Qdrant: {check_service('http://localhost:6333')}")


# Display the main application title and description
st.markdown(
    """
    <div style="text-align:center">
        <h1>🤖 VectorMind - RAG</h1>
        <p class="small-text">
        Llama 3.2 + Qdrant + LangGraph — 100% Local
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


# Document upload section - allows users to upload and index PDF documents
with st.expander("📂 Upload de Documento"):
    uploaded_file = st.file_uploader("Envie um PDF", type=["pdf"])

    if uploaded_file:
        if st.button("Indexar Documento"):
            with st.spinner("Indexando documento..."):
                # Send the uploaded file to the backend for indexing
                files = {"file": uploaded_file.getvalue()}
                response = requests.post(f"{API_URL}/upload", files=files)

                # Provide user feedback based on the response status
                if response.status_code == 200:
                    st.success("Documento indexado com sucesso.")
                    st.session_state["document_ready"] = True
                else:
                    st.error("Erro ao indexar documento.")
                    st.json(response.json())


# Create a chat container for displaying messages
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display all previous messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input from the chat input field
if prompt := st.chat_input("Pergunte algo sobre o documento..."):

    # Validate that a document has been indexed before processing questions
    if not st.session_state.get("document_ready", False):
        st.warning("Indexe um documento antes de fazer perguntas.")
        st.stop()

    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    # Display user message in the chat interface
    with st.chat_message("user"):
        st.markdown(prompt)

    # Create assistant response section with streaming effect
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        status_placeholder = st.empty()

        # Show status message
        status_placeholder.info("Analisando documento...")

        start_time = time.time()

        try:
            # Send the question to the backend API and get the response
            response = requests.post(
                f"{API_URL}/ask",
                params={"question": prompt},
                timeout=120
            )

            status_placeholder.empty()

            # Process successful response from the backend
            if response.status_code == 200:
                answer = response.json().get("answer", "")
                full_response = ""

                # Stream the response word by word for a typewriter effect
                for chunk in answer.split():
                    full_response += chunk + " "
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.015)

                # Display final response without the cursor
                message_placeholder.markdown(full_response)

                # Show the time it took to process the question
                elapsed = time.time() - start_time
                st.caption(f"⏱️ {elapsed:.2f}s")

                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )

            else:
                # Display error message if backend responds with an error
                message_placeholder.error("Erro ao consultar backend.")
                st.code(response.text)

        except requests.exceptions.RequestException as e:
            # Handle connection errors to the backend
            status_placeholder.empty()
            message_placeholder.error(f"Erro de conexão: {e}")

# Close the chat container div
st.markdown('</div>', unsafe_allow_html=True)
