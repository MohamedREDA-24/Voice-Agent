# Voice Agent with RAG

A voice-enabled AI agent powered by Gemini that uses Retrieval-Augmented Generation (RAG) to answer questions from a knowledge base.

> 📖 **For detailed technical documentation**, see [ARCHITECTURE.md](ARCHITECTURE.md) - includes architecture diagrams, RAG integration details, and advanced configuration.

## Features

- 🎤 Real-time voice conversation with AI agent
- 📚 Knowledge base integration using RAG
- 🔍 Semantic search over documents
- 🌐 Web UI for easy interaction
- 🎮 LiveKit Playground support
- 🎯 Strict knowledge base mode (only answers from provided documents)

## Architecture

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

**High-level overview:**
- User connects via Web UI or LiveKit Playground → LiveKit Server
- Agent processes audio with Gemini Live API
- RAG system provides knowledge base context
- Agent responds with audio using knowledge base information

## Quick Start

### 1. Setup Environment

Copy `env.example` to `.env` and fill in your credentials:

```bash
cp env.example .env
```

Required variables:
- `GOOGLE_API_KEY` - Get from https://ai.google.dev/
- `LIVEKIT_URL` - Your LiveKit server URL
- `LIVEKIT_API_KEY` - LiveKit API key
- `LIVEKIT_API_SECRET` - LiveKit API secret

### 2. Install Agent Dependencies

```bash
cd agent
pip install -r requirements.txt
```

### 3. Setup Knowledge Base

Add your documents to `knowledge_base/documents/`:
- `.txt` files
- `.pdf` files

The RAG system will automatically index them on first run.

### 4. Start the Agent

```bash
cd agent
python agent.py dev
```

The agent will automatically connect to LiveKit rooms as they are created.

### 5. Testing with LiveKit Playground

The easiest way to test the agent is using LiveKit Cloud's playground:

1. **Start the Agent First**:
   ```bash
   cd agent
   python agent.py dev
   ```
   Wait until you see "registered worker" in the logs. The agent is now waiting for rooms.

2. **Open LiveKit Cloud Playground**: 
   - Go to https://agents-playground.livekit.io/
   - Navigate to your project

3. **Join the Playground**: 
   - LiveKit Cloud automatically creates a room for you
   - Allow microphone access when prompted
   - The playground will show a room name like `playground-abc123`

4. **Agent Auto-Connects**: 
   - Your agent will automatically detect and join the room
   - Check agent logs for: `🎯 Joining room: [room-name]`
   - You should see: `✅ Session connected with knowledge base embedded`
   - The playground will show the agent as a participant

5. **Start Talking**:
   - Click the microphone button in the playground
   - Speak your question (e.g., "What is your return policy?")
   - The agent will respond using only the knowledge base
   - You'll hear the agent's voice response

**Tips**:
- Make sure the agent is running **before** joining the playground
- LiveKit Cloud auto-generates room names - the agent will connect automatically
- Check agent logs if you don't see the agent in the playground
- The agent uses the knowledge base embedded in its system prompt

### 6. Start Web UI (Optional)

The web UI allows you to interact with the agent through a browser interface.

**Note:** The token server is only needed if you want to use the web UI. The agent can run standalone without it.

Quick start:
```bash
# Terminal 1: Token server (required for web UI only)
cd server
pip install -r requirements.txt
python token_server.py

# Terminal 2: Web UI
cd web
npm install
npm run dev
```

Then open `http://localhost:3000` in your browser.

**Setup Options:**

1. **Agent + Playground** (Simplest):
   - Just run the agent: `cd agent && python agent.py dev`
   - Use LiveKit Playground in your browser to connect
   - No token server or web UI needed

2. **Agent + Custom Web UI**:
   - Run agent + token server + web UI
   - Full control over the UI experience

3. **Agent Only**:
   - Agent runs and waits for rooms to be created
   - Can be used with any LiveKit client

## Project Structure

```
Voice-Agent/
├── agent/              # Python agent code
│   ├── agent.py       # Main agent implementation
│   ├── rag.py        # RAG system
│   └── requirements.txt
├── web/               # React frontend (optional)
│   └── src/
├── server/            # Token server (for web UI)
│   └── token_server.py
└── knowledge_base/    # Documents and embeddings
    ├── documents/     # Your knowledge base files
    └── embeddings/   # Generated FAISS index
```

For detailed project structure and component descriptions, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Knowledge Base

The agent uses a RAG (Retrieval-Augmented Generation) system to answer questions from your knowledge base.

### Adding Documents

1. Place `.txt` or `.pdf` files in `knowledge_base/documents/`
2. The agent will automatically index them on startup (if index doesn't exist)
3. Documents are split into chunks and embedded (see [ARCHITECTURE.md](ARCHITECTURE.md) for details)

### Current Documents

- `faq.txt` - Frequently asked questions
- `product_info.txt` - Product information

For detailed information about RAG system, embeddings, and retrieval, see [ARCHITECTURE.md](ARCHITECTURE.md).

## How It Works

1. **User speaks** → Audio sent to LiveKit room
2. **Agent receives** → Processes audio with Gemini Live API
3. **Knowledge base** → Embedded in system prompt (RAG system)
4. **Response generation** → Gemini answers using knowledge base context
5. **Audio response** → Agent speaks back to user

For detailed data flow and RAG integration, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Development

### Testing RAG System

```bash
cd agent
python test_rag.py
```

### Rebuilding Knowledge Base

```bash
cd agent
python -c "from rag import RAGSystem; rag = RAGSystem(); rag.build_index(force_rebuild=True)"
```

For detailed local development setup and advanced usage, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Configuration

### Quick Configuration

- **Agent**: Edit `agent/agent.py` - `use_rag=True` to enable/disable RAG
- **RAG**: Edit `agent/rag.py` - `embedding_model="google"` or `"local"`

For detailed configuration options and explanations, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Troubleshooting

### Quick Fixes

**Agent not connecting:**
- Check LiveKit credentials in `.env` file
- Verify agent shows "registered worker" in logs
- Ensure agent is running before joining playground

**Knowledge base not working:**
- Check documents exist in `knowledge_base/documents/`
- Verify RAG index is built (check `knowledge_base/embeddings/faiss_index/`)
- Look for "✅ RAG ready" message in agent logs

**Web UI issues:**
- Token server must be running on port 8080
- Check `http://localhost:8080/api/health`
- Verify LiveKit credentials in `.env`

**Audio issues:**
- Check browser microphone permissions
- Ensure HTTPS or localhost for microphone access

For detailed troubleshooting and advanced debugging, see [ARCHITECTURE.md](ARCHITECTURE.md).

## License

MIT
