# Architecture & Technical Documentation

## System Architecture

### Overview

The Voice Agent system combines LiveKit for real-time audio communication, Google's Gemini Live API for voice processing, and a RAG (Retrieval-Augmented Generation) system for knowledge base integration.

```
┌──────────────┐
│   User       │
│  (Browser)   │
└──────┬───────┘
       │ Audio Stream
       ▼
┌─────────────────────────────────────────────────┐
│            LiveKit Server                       │
│  (WebSocket-based real-time communication)      │
└──────┬──────────────────────────────┬───────────┘
       │                              │
       │ Room Events                  │ Room Events
       ▼                              ▼
┌──────────────┐              ┌──────────────────┐
│  Web UI      │              │  Python Agent    │
│  (React)     │              │  (LiveKit SDK)   │
└──────────────┘              └────────┬─────────┘
                                       │
                                       │ Audio Processing
                                       ▼
                              ┌──────────────────┐
                              │  Gemini Live API │
                              │  (Native Audio)  │
                              └────────┬─────────┘
                                       │
                                       │ Context Retrieval
                                       ▼
                              ┌──────────────────┐
                              │   RAG System     │
                              │  (FAISS + Embed) │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ Knowledge Base    │
                              │  (Documents)     │
                              └──────────────────┘
```

## Component Details

### 1. LiveKit Server

**Role**: Real-time communication hub

- Manages WebSocket connections for audio streaming
- Routes audio between participants (user and agent)
- Handles room management and participant lifecycle
- Provides event system for connection/disconnection

**Connection Flow**:
1. User connects via Web UI or Playground → LiveKit room created
2. Agent worker registered with LiveKit → Waits for room assignments
3. When room is created → LiveKit dispatches job to agent
4. Agent joins room → Audio tracks published/subscribed
5. Bidirectional audio stream established

### 2. Python Agent (`agent/agent.py`)

**Role**: Voice processing and response generation

**Key Components**:

- **LiveKit Integration**: Uses `livekit-agents` SDK
  - Listens for room assignments via `JobContext`
  - Publishes audio output track
  - Subscribes to user's audio input track
  - Handles audio frame processing

- **Gemini Live API**: Uses `google-genai` SDK
  - Maintains persistent WebSocket session with Gemini
  - Sends audio chunks via `send_realtime_input()`
  - Receives audio responses via `session.receive()`
  - Handles transcription and model responses

- **RAG Integration**: Uses custom `RAGSystem` class
  - Retrieves relevant context before response generation
  - Embeds knowledge base in system prompt

### 3. RAG System (`agent/rag.py`)

**Role**: Knowledge base retrieval and embedding

**Components**:

- **Document Loading**: Reads `.txt` and `.pdf` files from `knowledge_base/documents/`
- **Text Splitting**: Uses LangChain's `RecursiveCharacterTextSplitter`
  - Chunk size: 500 characters
  - Overlap: 50 characters
  - Preserves document structure

- **Embedding Generation**:
  - **Primary**: Google's `text-embedding-004` (via API)
  - **Fallback**: Local `all-MiniLM-L6-v2` (sentence-transformers)
  - Prints which model is used during generation

- **Vector Storage**: FAISS (Facebook AI Similarity Search)
  - Index type: `IndexFlatIP` (Inner Product for cosine similarity)
  - Normalized embeddings for cosine similarity
  - Persistent storage: `index.faiss` + `metadata.pkl`

- **Retrieval**: Semantic search with reranking
  - Generates query embedding
  - Searches FAISS index
  - Returns top-k most relevant chunks
  - Filters by similarity threshold

### 4. Gemini Live API Integration

**How It Works**:

1. **Session Initialization**:
   ```python
   async with client.aio.live.connect(
       model="gemini-2.5-flash-native-audio-preview-09-2025",
       config={
           "response_modalities": ["AUDIO"],
           "system_instruction": system_instruction  # Contains knowledge base
       }
   ) as session:
   ```

2. **Audio Streaming**:
   - Agent receives audio frames from LiveKit
   - Converts to PCM format (16kHz, mono)
   - Sends to Gemini via `session.send_realtime_input(audio=...)`
   - Gemini processes audio in real-time

3. **Response Handling**:
   - Gemini generates audio responses
   - Agent receives via `session.receive()`
   - Extracts audio data from `model_turn.parts`
   - Plays audio through LiveKit audio track

## RAG Integration with Gemini Live API

### Knowledge Base Embedding Strategy

The system uses a **system prompt embedding** approach rather than per-query retrieval:

1. **At Agent Startup**:
   ```python
   # Load all knowledge base chunks
   self.knowledge_base_summary = self._prepare_knowledge_summary()
   
   # Embed in system instruction
   system_instruction = base_instruction + self.knowledge_base_summary
   ```

2. **System Prompt Structure**:
   ```
   You are a helpful voice assistant with access to a knowledge base.
   
   RULES:
   1. Answer questions using the knowledge base provided below
   2. Be conversational and natural
   ...
   
   KNOWLEDGE BASE:
   [Context 1 from faq.txt - relevance: 0.83]
   ## General Information
   ### What is your return policy?
   We offer a 30-day return policy...
   
   [Context 2 from product_info.txt - relevance: 0.70]
   ...
   ```

3. **Why This Approach**:
   - **Simplicity**: Single system prompt, no per-query API calls
   - **Speed**: No retrieval latency during conversation
   - **Context**: Full knowledge base always available to model
   - **Consistency**: Model has complete context for reference resolution


## Data Flow

### Complete Request-Response Cycle

1. **User Speaks**:
   ```
   Browser → LiveKit Server → Agent (audio frames)
   ```

2. **Audio Processing**:
   ```
   Agent receives PCM audio → Sends to Gemini Live API
   ```

3. **Gemini Processing**:
   ```
   Gemini transcribes audio → Generates response using system prompt
   (System prompt contains full knowledge base)
   ```

4. **Response Generation**:
   ```
   Gemini generates audio response → Sends back to Agent
   ```

5. **Audio Playback**:
   ```
   Agent receives audio → Publishes to LiveKit → User hears response
   ```

### RAG Retrieval Flow (If Using Per-Query)

1. **Transcription Detection**:
   ```python
   if hasattr(server_content, "input_transcription"):
       query = server_content.input_transcription.text
   ```

2. **Context Retrieval**:
   ```python
   # Generate query embedding
   query_embedding = rag._generate_embedding(query)
   
   # Search FAISS index
   distances, indices = rag.index.search(query_embedding, top_k=5)
   
   # Retrieve relevant chunks
   context = [rag.metadata[i]['content'] for i in indices]
   ```

3. **Context Injection**:
   ```python
   # Send to Gemini as text input
   await session.send_text(f"[CONTEXT]\n{context}\n[QUESTION]\n{query}")
   ```

## Local Development Setup

### Prerequisites

- Python 3.8+
- Node.js 18+ (for web UI)
- LiveKit Cloud account or self-hosted server
- Google API key for Gemini

### Step-by-Step Setup

#### 1. Environment Configuration

Create `.env` file in project root:

```env
# Google API
GOOGLE_API_KEY=your_google_api_key_here

# LiveKit (from https://cloud.livekit.io)
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
```

#### 2. Install Agent Dependencies

```bash
cd agent
pip install -r requirements.txt
```

**Key Dependencies**:
- `livekit-agents==1.3.2` - LiveKit agent framework
- `google-genai==1.50.1` - Gemini API client
- `faiss-cpu==1.13.0` - Vector similarity search
- `langchain-core`, `langchain-community` - Document processing
- `sentence-transformers` - Local embedding fallback

#### 3. Setup Knowledge Base

```bash
# Add documents
cp your-documents.txt knowledge_base/documents/

# Index will be built automatically on first run
# Or manually:
cd agent
python -c "from rag import RAGSystem; rag = RAGSystem(); rag.build_index()"
```

#### 4. Run the Agent

```bash
cd agent
python agent.py dev
```

**What Happens**:
1. Agent initializes RAG system
2. Loads or builds FAISS index
3. Prepares knowledge base summary
4. Registers with LiveKit as worker
5. Waits for room assignments

**Expected Output**:
```
============================================================
🔧 INITIALIZING RAG SYSTEM
============================================================
✅ Loaded existing FAISS index with 25 chunks
✅ RAG system initialized
✅ RAG ready with 25 chunks
✅ Prepared knowledge base summary (6057 chars)
============================================================
INFO   livekit.agents   registered worker
```

#### 5. Test with LiveKit Playground

1. Go to https://agents-playground.livekit.io/
2. Join room (auto-created)
3. Agent automatically connects
4. Start speaking

#### 6. Optional: Run Web UI

```bash
# Terminal 1: Token Server
cd server
pip install -r requirements.txt
python token_server.py

# Terminal 2: Web UI
cd web
npm install
npm run dev
```

Open http://localhost:3000

## Configuration Options

### Agent Configuration (`agent/agent.py`)

```python
class GeminiVoiceAgentWithRAG:
    def __init__(self, use_rag: bool = True):
        self.use_rag = use_rag  # Enable/disable RAG
        self.model = "gemini-2.5-flash-native-audio-preview-09-2025"
```

### RAG Configuration (`agent/rag.py`)

```python
class RAGSystem:
    def __init__(
        self,
        chunk_size: int = 500,      # Text chunk size
        chunk_overlap: int = 50,    # Overlap between chunks
        embedding_model: str = "google"  # "google" or "local"
    ):
```

### System Prompt Customization

Edit `agent/agent.py` → `_run_continuous_session()`:

```python
base_instruction = """You are a helpful voice assistant...
RULES:
1. Answer questions using the knowledge base
2. Be conversational
...
"""
```

## Troubleshooting

### Agent Not Connecting

- Check LiveKit credentials in `.env`
- Verify agent shows "registered worker" in logs
- Ensure LiveKit URL is correct (wss:// protocol)

### RAG Not Working

- Check `knowledge_base/embeddings/faiss_index/` exists
- Verify documents in `knowledge_base/documents/`
- Look for "RAG ready" message in logs
- Check system prompt length (should include knowledge base)

### Audio Issues

- Verify microphone permissions
- Check browser console for errors
- Ensure LiveKit server is accessible
- Check agent logs for audio track messages

### Embedding Model Issues

- Google API: Check `GOOGLE_API_KEY` is set
- Local fallback: Ensure `sentence-transformers` installed
- Check logs for which model is being used

## Performance Considerations

### System Prompt Size

- Current: ~6000 characters (20 chunks)
- Gemini limit: ~1M tokens
- Trade-off: More chunks = better coverage, but larger prompt

### Embedding Generation

- Google API: ~100-200ms per chunk
- Local model: ~10-50ms per chunk (after initial load)
- Index building: ~2-5 seconds for 25 chunks

### Retrieval Speed

- FAISS search: <1ms for 25 vectors
- Query embedding: Same as chunk embedding
- Total retrieval: <200ms typically

## Future Enhancements

### Potential Improvements

1. **Hybrid Retrieval**: Combine system prompt + per-query retrieval
2. **Streaming RAG**: Update context during conversation
3. **Multi-modal**: Add image/document support
4. **Caching**: Cache frequent queries
5. **Fine-tuning**: Fine-tune embeddings on domain data

## References

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [LangChain Documentation](https://python.langchain.com/)

