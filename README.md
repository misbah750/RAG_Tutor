## RAG PDF Chatbot (Streamlit + LangChain + Chroma)

A modern, dark-themed Streamlit app to chat with your PDFs using Retrieval-Augmented Generation (RAG).

Built by: Misbah Ul Hasan, Abdul Ali Zaidi

### Features
- Upload a PDF and ask questions answered strictly from the document
- Sources include page number and best-effort section/heading
- Deduplicated, sorted source references (e.g., "Page 12 — 1.1 Background")
- Local models via Ollama, with optional OpenAI fallback
- Persistent Chroma DB per session (auto-cleaned on new upload)
- Optimized for Windows/PowerShell; caching added to reduce lag

### Requirements
- Python 3.12 (recommended)
- Windows PowerShell
- Dependencies listed in `requirements.txt`
- One of the following for LLM/embeddings:
  - Ollama (local inference): `llama3.2` (chat) and `mxbai-embed-large` (embeddings)
  - Optional: OpenAI via `OPENAI_API_KEY` (enables `gpt-3.5-turbo` and OpenAI embeddings)

### Installation (Windows PowerShell)
```powershell
cd "C:\Users\misba\OneDrive\Desktop\New folder (2)"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If using Ollama (recommended for local):
```powershell
# Install Ollama if needed: https://ollama.com/download
ollama --version
ollama pull llama3.2
ollama pull mxbai-embed-large
```

### Run the app
```powershell
python -m streamlit run App.py --server.port=8505 --server.headless true
```
Open `http://localhost:8505` in your browser.

If the port is already in use, change `--server.port` (e.g., 8506).

### Environment Variables (optional)
- `OPENAI_API_KEY`: If set, the app exposes `gpt-3.5-turbo` and OpenAI embeddings.

Create a `.env` file in the project root to load variables automatically:
```
OPENAI_API_KEY=sk-...
```

### How It Works (High Level)
1. PDF is uploaded and stored temporarily
2. Text is loaded via `PyPDFLoader` and split into overlapping chunks
3. Chunks are embedded (Ollama `mxbai-embed-large` or OpenAI embeddings)
4. Chroma vector store is created and persisted for the session
5. Questions are answered by `ConversationalRetrievalChain`, restricted to document context
6. Sources are returned with page and detected section heading; duplicates are removed and sorted

### Model Selection
- Sidebar shows only locally installed Ollama models to avoid 404s
- If `OPENAI_API_KEY` is present, `gpt-3.5-turbo` appears as an option
- Current default local chat model: `llama3.2`
- Current embedding model: `mxbai-embed-large`

### Using other models (extensible)
- You can add more Ollama models (e.g., `qwen2.5`, `mistral`) by pulling them:
  ```powershell
  ollama pull qwen2.5:0.5b
  ollama pull mistral
  ```
- Then expose them in the sidebar by adding their names to the list in `App.py` near `handle_sidebar()`.
- For OpenAI, set `OPENAI_API_KEY` and change the `selected_model` to a supported model name.

### Common Commands
- Restart on a new port if the current port is busy:
```powershell
python -m streamlit run App.py --server.port=8506 --server.headless true
```
- Clear app cache from the sidebar using "Clear Cache"
- Clear chat and index (Chroma) using "Clear Chat"

### Troubleshooting
- "model not found (404)":
  - Run `ollama pull llama3.2` and `ollama pull mxbai-embed-large`
- Streamlit not found: Use `python -m streamlit ...`
- Port already in use: change `--server.port` value
- App slow/lagging:
  - Ensure Ollama is running and models are pulled
  - Close other heavy processes; try a smaller PDF
  - Network filters/firewall can slow initial model start
- ValidationError for `keep_alive`:
  - The app uses an integer (seconds). If you still see the error, fully restart Streamlit so caches refresh
  - If an old error references `"5m"`, clear cache via the sidebar or restart the app

### Project Structure
```
App.py               # Streamlit app
requirements.txt     # Python deps
db/                  # Chroma persistence (ephemeral per upload)
```

### Notes on Accuracy
- Chunking is tuned (size 1200, overlap 250) to preserve section coherence
- The QA prompt is strict: if content isn’t in the document, it replies "Not found in document."
- Headings are extracted heuristically; exact chapter detection depends on PDF quality

### License
This project is provided as-is for educational use.


