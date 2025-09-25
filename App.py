#To install all the required libraries, run this command in your terminal:
#pip install streamlit python-dotenv langchain langchain-ollama langchain-openai chromadb pypdf


import os
import re
import shutil
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_vectorstore_agent
import subprocess
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

load_dotenv() # load environment variables


#--------------------- App Configuration ---------------------
def configure_page():
    """Configure the Streamlit page settings"""
    # Set the page title, icon,layout and initial sidebar state
    st.set_page_config(
        page_title='RAG PDF Chatbot',
        page_icon='📚',
        layout='wide',
        initial_sidebar_state='expanded'
    )
    # Global style overrides for a more polished, professional UI
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        :root {
            --brand-500: #6366f1; /* indigo */
            --brand-400: #818cf8;
            --brand-600: #4f46e5;
            --bg-50: #f8fafc;
            --bg-100: #f1f5f9;
            --text-900: #0f172a;
            --text-700: #334155;
            --card-bg: #ffffff;
            --border: #e5e7eb;
        }
        html, body, [data-testid="stAppViewContainer"] * { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
        /* Dark theme */
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            color: #f1f5f9;
        }
        /* Main content area */
        .main .block-container {
            background: rgba(15, 23, 42, 0.8);
            border-radius: 20px;
            padding: 2rem;
            margin-top: 1rem;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        /* Hide default Streamlit header space */
        [data-testid="stHeader"] { background: transparent; }
        /* Sidebar visuals */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] a { color: #c7d2fe !important; }
        [data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        /* Cards and containers */
        .subtle-card { 
            background: rgba(30, 41, 59, 0.8); 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            border-radius: 14px; 
            padding: 16px; 
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            color: #f1f5f9;
        }
        .muted { color: #94a3b8; }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, var(--brand-600), var(--brand-400));
            color: #fff; border: 0; border-radius: 12px; padding: 0.6rem 1rem;
            box-shadow: 0 10px 24px rgba(79,70,229,0.25);
            transition: transform .06s ease, filter .2s ease;
        }
        .stButton>button:hover { filter: brightness(1.07); transform: translateY(-1px); }
        .stButton>button:active { transform: translateY(0px) scale(0.99); }
        /* File uploader */
        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.6); 
            padding: 16px; 
            border-radius: 14px; 
            border: 1px dashed rgba(255, 255, 255, 0.3);
            color: #f1f5f9;
        }
        /* Chat bubbles */
        .chat-bubble { padding: 12px 16px; border-radius: 16px; margin: 8px 0; line-height: 1.6; }
        .user-bubble { 
            background: linear-gradient(135deg, #1e293b, #334155); 
            color: #f8fafc; 
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.3) inset, 0 6px 18px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .assistant-bubble { 
            background: rgba(30, 41, 59, 0.9); 
            color: #f1f5f9; 
            border: 1px solid rgba(255, 255, 255, 0.1); 
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
        }
        .source-badge { 
            display: inline-block; 
            padding: 3px 10px; 
            border-radius: 999px; 
            background: rgba(99, 102, 241, 0.2); 
            color: #c7d2fe; 
            border: 1px solid rgba(99, 102, 241, 0.3); 
            font-size: 12px; 
            margin-right: 6px; 
        }
        /* Chat input spacing */
        [data-testid="stChatInput"] { margin-top: .5rem; }
        /* Divider spacing */
        hr { border-color: rgba(255, 255, 255, 0.2); opacity: .6; }
        /* Text inputs */
        .stTextInput>div>div>input {
            background: rgba(30, 41, 59, 0.8);
            color: #f1f5f9;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stTextInput>div>div>input:focus {
            border-color: var(--brand-500);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }
        /* Selectbox */
        .stSelectbox>div>div {
            background: rgba(30, 41, 59, 0.8);
            color: #f1f5f9;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Hero header
    st.markdown("<h1 style='margin-bottom:0'>💡 Chat with your PDF</h1>", unsafe_allow_html=True)
    st.markdown("<p class='muted' style='margin-top:6px'>Ask questions about your documents and cite exact pages.</p>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize the session state variables"""
    # Initialize messages history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Initialize conversation chain
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    # Initialize PDF processing status
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = None
    # Initialize vector store persistence directory
    if "persist_directory" not in st.session_state:
        st.session_state.persist_directory = None
    # Initialize default model
    if "model" not in st.session_state:
        # Prefer OpenAI if API key exists to avoid requiring local Ollama by default
        st.session_state.model = "gpt-3.5-turbo" if os.getenv("OPENAI_API_KEY") else "llama3.2"

#--------------------- Model Setup ---------------------
@st.cache_resource
def get_chat_model(model_name):
    """Get the chat model based on selected model name"""
    if model_name == "gpt-3.5-turbo" and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model_name,
            streaming=True,
        )
    # Fallback to Ollama if OpenAI is not configured
    return ChatOllama(
        model=model_name,
        streaming=True,
        temperature=0.1,
        num_ctx=4096,
    )

def get_embeddings(version: int = 3):
    """Get the embeddings model for processing PDF"""
    # Use OpenAI embeddings if API key is available; otherwise fall back to Ollama
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return OllamaEmbeddings(model="mxbai-embed-large", keep_alive=300)


#--------------------- PDF Processing ---------------------
def process_pdf(pdf_file):
    """Process the PDF and create a vector store"""
    # Save the uploaded PDF to temp file
    tmp_file_path = save_temp_file(pdf_file)
    # load doc from temp pdf file
    documents = load_pdf(tmp_file_path)
    # split the loaded document into chunks
    chunks = split_pdf(documents)
    # Create the directory for persisting the vector store
    persist_directory = create_chroma_persist_directory()
    # Create a vector store from chunks
    vectorstore = create_vectorstore(chunks,persist_directory)
    # Remove temp file after processing
    os.unlink(tmp_file_path)
    # Save simple metrics for the UI
    try:
        st.session_state.num_pages = len(documents)
    except Exception:
        st.session_state.num_pages = None
    st.session_state.num_chunks = len(chunks) if chunks else 0
    return vectorstore


def save_temp_file(pdf_file):
    """Save the uploaded PDF file to temp file and return the file path"""
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        return tmp_file.name


@st.cache_data(show_spinner=False)
def load_pdf(tmp_file_path):
    """load the PDF file and retuen the documents"""
    loader = PyPDFLoader(tmp_file_path)
    return loader.load()

@st.cache_data(show_spinner=False)
def split_pdf(documents):
    """Split the pdf doc into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # slightly larger chunks preserve section coherence
        chunk_overlap=250,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        if "page" not in chunk.metadata and "page" in documents[0].metadata:
            chunk.metadata["page"] = documents[0].metadata["page"]
    return chunks

def extract_heading(text):
    """Best-effort heading/section extraction from a chunk of text."""
    if not text:
        return None
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        # Match patterns like "Chapter 1", "1.1 Background", "2 Introduction"
        if re.match(r"^(Chapter\s+\d+\b.*|\d+(?:\.\d+)*\s+.+)$", candidate, re.IGNORECASE):
            return candidate[:100]
    return None

def create_chroma_persist_directory():
    """ Create a directory for persisting vector store"""
    persist_directory = 'db'
    st.session_state.persist_directory = persist_directory
    return persist_directory

def create_vectorstore(chunks,persist_directory):
    """ Create Chroma vector store from doc chunks"""
    embedding_model = get_embeddings(version=3)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )


# --------------------- Chat Interface ---------------------
def initialize_conversation(vectorstore, chat_model):
    """Initialize a conversational retrieval chain with the given vector store and chat model"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",  # Ensure the chain knows which output to store in memory
    )

    # Strict prompt: answer only from the provided context
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant for question-answering tasks.\n"
            "Only use the following document excerpts to answer the question.\n"
            "If the answer is not contained in the excerpts, reply exactly: 'Not found in document.'\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
    )

    # Initialize a conversational retrieval chain with given parameters
    return ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        ),
        memory=memory,
        verbose=False,
        return_source_documents=True,   # This ensures we also get PDF page metadata
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

def display_chat_message():
    """Display the chat messages in chat interface"""
    # Iterate through each message in the chat history
    for message in st.session_state.messages:
        # check if the message is from user
        if isinstance(message,HumanMessage):
            with st.chat_message("user"):
                st.markdown(f"<div class='chat-bubble user-bubble'>{message.content}</div>", unsafe_allow_html=True)
        # check if the message is from Assistant
        if isinstance(message,AIMessage):
            with st.chat_message("assistant"):
                st.markdown(f"<div class='chat-bubble assistant-bubble'>{message.content}</div>", unsafe_allow_html=True)
        # check if the message is sys message
        if isinstance(message,SystemMessage):
            with st.chat_message("system"):
                st.markdown(f"<div class='chat-bubble assistant-bubble'>{message.content}</div>", unsafe_allow_html=True)


def handle_user_input(conversation):
    """Handle user input and chat interactions with the assistant."""
    # Get user input from the chat input widget
    if prompt := st.chat_input("Ask questions about your PDF:"):
        # Create a HumanMessage instance with the input prompt and append it to session state
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)

        # Display the user message in the chat
        with st.chat_message("user"):
            st.write(prompt)

        # Assistant response bubble
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            try:
                # Get the response from the conversation model
                response = conversation.invoke({"question": prompt})
                answer = response.get("answer", "I'm not sure how to respond to that.")

                # ✅ Extract sources (real PDF page numbers)
                sources = response.get("source_documents", [])
                if sources:
                    unique_labels = []
                    seen = set()
                    for doc in sources:
                        page_number = doc.metadata.get("page")
                        heading = extract_heading(getattr(doc, "page_content", ""))
                        # Use page number as primary key for de-duplication
                        key = (page_number, heading)
                        if key in seen:
                            continue
                        seen.add(key)
                        if page_number is not None:
                            label = f"Page {page_number + 1}"
                        else:
                            label = "Unknown page"
                        if heading:
                            label += f" — {heading}"
                        unique_labels.append((page_number if page_number is not None else 10**9, label))
                    # sort by page number when available
                    unique_labels.sort(key=lambda x: x[0])
                    answer += "\n\n📄 **Sources:** " + ", ".join(label for _, label in unique_labels)

            except Exception as e:
                answer = f"Error: {e}"

            # Show the assistant's final answer (with sources)
            message_placeholder.markdown(f"<div class='chat-bubble assistant-bubble'>{answer}</div>", unsafe_allow_html=True)

        # Create an AIMessage instance with the answer and append it to session state
        assistant_message = AIMessage(content=answer)
        st.session_state.messages.append(assistant_message)

        # (Follow-up box removed per user request)


#--------------------- Sidebar Functions ---------------------
def handle_sidebar():
    """handle the sidebar interactions and return the selected model name"""
    st.sidebar.markdown("### ⚙️ Settings")
    
    # discover installed ollama models to avoid 404 errors
    installed_ollama_models = set()
    try:
        output = subprocess.check_output(["ollama","list"], text=True)
        for line in output.splitlines():
            if line and not line.startswith("NAME"):
                name = line.split()[0]
                installed_ollama_models.add(name)
    except Exception:
        pass

    available_models = []
    # Show OpenAI if available
    if os.getenv("OPENAI_API_KEY"):
        available_models.append("gpt-3.5-turbo")
    # Add Ollama models only if present locally
    for m in ("llama3.2","llama3.2:1b","qwen2.5:0.5b"):
        if m in installed_ollama_models:
            available_models.append(m)
    # Fallback in case list ends up empty
    if not available_models:
        available_models = ["llama3.2"]

    selected_model = st.sidebar.selectbox(
        "Select a Model", available_models
    )
    st.session_state.model = selected_model
    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear Chat"):
        clear_chat()
    if st.sidebar.button("♻️ Clear Cache"):
        clear_cache()
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Model")
    st.sidebar.markdown(f"<div class='subtle-card'>Current: <b>{selected_model}</b></div>", unsafe_allow_html=True)
    st.sidebar.markdown("#### About")
    st.sidebar.markdown("RAG PDF Chatbot with Streamlit, LangChain, and Chroma.")
    st.sidebar.markdown("#### Developers")
    st.sidebar.markdown("**Misbah Ul Hasan**")
    st.sidebar.markdown("**Abdul Ali Zaidi**")
    return selected_model
def clear_chat():
    """clear the chat history"""
    st.session_state.messages = []
    st.session_state.conversation = None
    cleanup_chroma_db()
    st.rerun()
    
def clear_cache():
    """ clear the streamlit cache to reset the application state"""
    st.cache_data.clear()
    st.cache_resource.clear()
    
def cleanup_chroma_db():
    """cleanup the chroma database"""
    persist_directory = st.session_state.get("persist_directory")
    
    #check if the persist directory exists
    if persist_directory and os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
            st.session_state.persist_directory = None
        except Exception as e:
            st.error(f"Error cleaning up chroma database: {e}")
    
#--------------------- PDF Upload Handler ---------------------
def handle_pdf_upload(pdf_file,chat_model):
    """Handle the PDF upload and process the PDF"""
    #check if the PDF has already been processed
    if st.session_state.pdf_processed != pdf_file.name:
        # Display a processing message while the PDF is being processed
        with st.spinner("Processing PDF..."):
            # Cleanup the previous Chroma database if it exists
            cleanup_chroma_db()
            # Process the uploaded PDF to create vector store
            vectorstore = process_pdf(pdf_file)
            # Initialize a new chat with the vector store and chat model
            st.session_state.conversation = initialize_conversation(
                vectorstore,chat_model
            )
            # Mark the PDF as processed in session state
            st.session_state.pdf_processed = pdf_file.name
            # Reset the messages in the session state
            st.session_state.messages = []
            st.success("PDF processed successfully!")

#--------------------- Main Application ---------------------
def main():
    """Main function to run the application"""
    configure_page()
    #initialize the session state
    initialize_session_state()
    #handle the sidebar interactions and get the selected model
    selected_model = handle_sidebar()
    #Get the chat model based on the selected model
    chat_model = get_chat_model(selected_model)
    # Layout: two columns
    left, right = st.columns([2, 1], gap="large")

    with left:
        pdf_file = st.file_uploader("Upload a PDF file",type="pdf")
        if pdf_file:
            handle_pdf_upload(pdf_file,chat_model)
        else:
            st.info("Please upload a PDF file to start chatting.")

        # Chat area
        st.markdown("---")
        display_chat_message()
        if st.session_state.conversation is not None:
            handle_user_input(st.session_state.conversation)

    with right:
        st.markdown("#### Document & Session", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("- **Model**: " + st.session_state.get("model", "-"))
            np = st.session_state.get("num_pages")
            nc = st.session_state.get("num_chunks")
            if np is not None:
                st.markdown(f"- **Pages**: {np}")
            if nc is not None:
                st.markdown(f"- **Chunks**: {nc}")
        st.markdown("#### Tips")
        st.markdown("- Ask for specific sections.\n- Use follow-ups to refine answers.\n- Look at cited pages.")

if __name__ == "__main__":
    main()
    



# To run program, use this command in your terminal:
# streamlit run <your_file_name>.py
