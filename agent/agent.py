# agent_with_rag_strict.py - Agent responds ONLY from RAG knowledge base
import asyncio
import os
import time
from typing import Optional, List, Dict
import numpy as np
from dotenv import load_dotenv
from google import genai
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
import pathlib

# Import RAG system
from rag import RAGSystem

# Load environment variables
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
for env_path in [script_dir / ".env", project_root / ".env.local", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break


class GeminiVoiceAgentWithRAG:
    def __init__(self, use_rag: bool = True, max_history: int = 10):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash-native-audio-preview-09-2025"
        self.use_rag = use_rag
        self.max_history = max_history
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Initialize RAG system
        if self.use_rag:
            print("🔧 Initializing RAG system...")
            self.rag = RAGSystem(embedding_model="google")
            if self.rag.index is None:
                print("📚 Building RAG index (first time setup)...")
                self.rag.build_index()
            print(f"✅ RAG ready with {len(self.rag.metadata)} chunks")
        
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None
        self._audio_buffer: List[bytes] = []
        self._buffer_threshold = 3200
        self._last_send_time = 0.0
        self._current_session = None
        self._is_running = True
        self._room = None
        
        # Track current turn state
        self._current_user_query = None
        self._context_sent_for_query = None
        self._current_agent_response = ""

    def _add_to_history(self, role: str, content: str):
        """Add message to conversation history with size limit"""
        self.conversation_history.append({"role": role, "content": content})
        
        # Keep only the last N exchanges (user + assistant pairs)
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-(self.max_history * 2):]
        
        print(f"📝 History size: {len(self.conversation_history)//2} exchanges")

    def _format_history_for_context(self) -> str:
        """Format conversation history as context"""
        if not self.conversation_history:
            return ""
        
        formatted = "\n\nPrevious conversation:\n"
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n"
        
        return formatted

    async def _setup_audio_track(self, room: rtc.Room):
        """Create and publish audio track"""
        self._audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        self._audio_track = rtc.LocalAudioTrack.create_audio_track("agent_audio", self._audio_source)
        
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        await room.local_participant.publish_track(self._audio_track, options)
        print("📡 Audio track published")

    async def process_conversation(self, ctx: JobContext):
        await ctx.connect()
        self._room = ctx.room
        print(f"🔌 Connected to room: {self._room.name}")

        await self._setup_audio_track(self._room)

        # Start session first
        session_task = asyncio.create_task(self._maintain_session())
        
        # Setup audio track handler
        audio_task = None
        
        # Wait for participant to join and track to be available
        @self._room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, *_):
            nonlocal audio_task
            if track.kind == rtc.TrackKind.KIND_AUDIO and audio_task is None:
                print(f"🎤 Audio track subscribed")
                audio_task = asyncio.create_task(self._handle_audio_track(track))
        
        # Check existing tracks
        for participant in self._room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.track:
                    print(f"🎤 Found existing audio track")
                    audio_task = asyncio.create_task(self._handle_audio_track(publication.track))
                    break
        
        try:
            # Keep the agent running - only wait for session_task
            if audio_task:
                await asyncio.gather(session_task, audio_task)
            else:
                await session_task
        except asyncio.CancelledError:
            print("🛑 Shutting down gracefully...")
        finally:
            self._is_running = False
            if audio_task and not audio_task.done():
                audio_task.cancel()
            if session_task and not session_task.done():
                session_task.cancel()
            print("🔚 Agent stopped")

    async def _maintain_session(self):
        """Maintain a continuous session with automatic reconnection"""
        while self._is_running:
            try:
                await self._run_continuous_session()
            except Exception as e:
                print(f"⚠️ Session error: {e}")
                if self._is_running:
                    print("🔄 Reconnecting in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    break

    async def _run_continuous_session(self):
        """Run ONE continuous session that persists across turns"""
        # STRICT system instruction - ONLY use knowledge base
        system_instruction = """You are a helpful knowledge base assistant. Follow these rules:

IMPORTANT RULES:
1. Answer questions using ONLY the information from [CONTEXT FROM KNOWLEDGE BASE]
2. The context will be provided with each question - read it carefully
3. If the context contains relevant information, answer confidently and naturally
4. Only say you don't have information if the context is truly empty or completely irrelevant
5. You may use [PREVIOUS CONVERSATION] to understand what "it", "that", "the previous one" refers to
6. Be conversational and helpful - you're speaking, not writing
7. Keep responses concise but complete

Example:
- Context has info about topic → Answer naturally using that info
- Context is empty or unrelated → "I don't have that information in my knowledge base"
- User asks "what about X?" after discussing Y → Check if context has X info, or if they mean Y from history"""

        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": system_instruction
        }
        
        print("🔄 Starting continuous session (STRICT RAG MODE)...")
        
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            self._current_session = session
            print("✅ Session connected - responding ONLY from knowledge base!")
            
            # Main loop - continuously handle responses
            while self._is_running:
                try:
                    async for response in session.receive():
                        if not self._is_running:
                            break
                        await self._handle_response(response)
                except Exception as e:
                    print(f"⚠️ Response handling error: {e}")
                    if self._is_running:
                        # Don't break - continue receiving
                        await asyncio.sleep(0.1)
                    else:
                        break

    async def _handle_response(self, response):
        """Handle each response in the continuous session"""
        if not hasattr(response, "server_content") or not response.server_content:
            return
        
        server_content = response.server_content
        
        # 1. Detect user speech and transcription
        if hasattr(server_content, "user_content") and server_content.user_content:
            for part in server_content.user_content.parts:
                if hasattr(part, "text") and part.text:
                    self._current_user_query = part.text
                    print(f"👤 User: {self._current_user_query}")
        
        # 2. When we have a NEW transcription, inject context
        if self._current_user_query and self._current_user_query != self._context_sent_for_query:
            await self._inject_context_for_query(self._current_user_query)
            self._context_sent_for_query = self._current_user_query
            self._current_agent_response = ""
        
        # 3. Handle model's audio response
        if hasattr(server_content, "model_turn") and server_content.model_turn:
            for part in server_content.model_turn.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    await self._play_audio(part.inline_data.data)
                
                if hasattr(part, "text") and part.text:
                    print(f"🤖 Agent: {part.text}")
                    self._current_agent_response += part.text + " "
        
        # 4. Check for turn completion
        if hasattr(server_content, "turn_complete") and server_content.turn_complete:
            # Add both messages to history
            if self._current_user_query:
                self._add_to_history("user", self._current_user_query)
            
            if self._current_agent_response.strip():
                self._add_to_history("assistant", self._current_agent_response.strip())
            
            print(f"✅ Turn complete - waiting for next interaction...\n")
            
            # Reset turn state but KEEP SESSION ALIVE
            self._current_user_query = None
            self._context_sent_for_query = None
            self._current_agent_response = ""

    async def _inject_context_for_query(self, query: str):
        """Inject RAG context and conversation history for current query"""
        enriched_parts = []
        
        # Add RAG context - REQUIRED for strict mode
        if self.use_rag:
            rag_context = await self._retrieve_context(query, top_k=5)
            if rag_context and rag_context.strip():
                print(f"📚 Injecting RAG context ({len(rag_context)} chars)...")
                enriched_parts.append(f"[CONTEXT FROM KNOWLEDGE BASE]\n{rag_context}\n[END OF CONTEXT]")
            else:
                print(f"⚠️ No context found - agent will say it doesn't know")
                enriched_parts.append("[CONTEXT FROM KNOWLEDGE BASE]\n(No information available)\n[END OF CONTEXT]")
        
        # Add conversation history for reference resolution
        if self.conversation_history:
            history_context = self._format_history_for_context()
            enriched_parts.append(f"[PREVIOUS CONVERSATION]{history_context}")
        
        # Add current user question
        enriched_parts.append(f"[USER QUESTION]\n{query}")
        
        # Combine and send
        enriched_message = "\n\n".join(enriched_parts)
        
        # Debug: print what we're sending
        print(f"📤 Sending to model:\n{enriched_message[:300]}...")
        
        await self._current_session.send_text(enriched_message)

    async def _retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context from RAG system"""
        if not self.use_rag:
            return ""
        
        try:
            print(f"🔍 Searching knowledge base for: '{query}'")
            loop = asyncio.get_event_loop()
            context = await loop.run_in_executor(
                None, 
                self.rag.retrieve_context, 
                query, 
                top_k
            )
            
            if context and context.strip():
                # Show first 200 chars for debugging
                preview = context[:200].replace('\n', ' ')
                print(f"   ✅ Found {len(context)} chars: {preview}...")
                return context
            else:
                print(f"   ❌ No relevant context found in knowledge base")
                return ""
                
        except Exception as e:
            print(f"⚠️ RAG error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    async def _handle_audio_track(self, track: rtc.Track):
        """Process incoming audio continuously"""
        audio_stream = rtc.AudioStream(track)
        print("🎧 Processing audio stream")
        
        try:
            async for event in audio_stream:
                if not self._is_running:
                    break
                    
                frame = event.frame
                pcm_data = self._audio_frame_to_pcm(frame)
                if pcm_data and self._current_session:
                    await self._send_audio_to_gemini(pcm_data)
        except asyncio.CancelledError:
            print("🎧 Audio stream handler cancelled")
        except Exception as e:
            print(f"❌ Audio error: {e}")

    async def _send_audio_to_gemini(self, pcm_data: bytes):
        """Buffer and send audio"""
        if not self._current_session:
            return
            
        self._audio_buffer.append(pcm_data)
        
        current_time = time.time()
        total_bytes = sum(len(c) for c in self._audio_buffer)
        
        if (total_bytes >= self._buffer_threshold or 
            (current_time - self._last_send_time) >= 0.1) and self._audio_buffer:
            
            combined = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            
            try:
                await self._current_session.send_realtime_input(
                    audio=genai.types.Blob(data=combined, mime_type="audio/pcm;rate=16000")
                )
                self._last_send_time = time.time()
            except Exception:
                pass

    def _audio_frame_to_pcm(self, frame: rtc.AudioFrame) -> Optional[bytes]:
        """Convert frame to PCM"""
        try:
            data = frame.data
            if isinstance(data, memoryview):
                data = data.tobytes()
            
            if isinstance(data, bytes):
                audio_array = np.frombuffer(data, dtype=np.int16)
            elif isinstance(data, np.ndarray):
                audio_array = data.astype(np.int16) if data.dtype != np.int16 else data
            else:
                return None

            if frame.sample_rate != 16000:
                downsample_factor = frame.sample_rate // 16000
                audio_array = audio_array[::downsample_factor]

            return audio_array.tobytes()
        except Exception:
            return None

    async def _play_audio(self, audio_data: bytes):
        """Play audio response"""
        try:
            if not self._audio_source:
                return

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            chunk_size = 480
            
            for i in range(0, len(audio_array), chunk_size):
                if not self._is_running:
                    break
                    
                chunk = audio_array[i:i + chunk_size]
                
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                frame = rtc.AudioFrame(
                    data=chunk.tobytes(),
                    sample_rate=24000,
                    num_channels=1,
                    samples_per_channel=len(chunk)
                )
                
                await self._audio_source.capture_frame(frame)
                await asyncio.sleep(0.001)
                
        except Exception as e:
            print(f"❌ Playback error: {e}")


async def entrypoint(ctx: JobContext):
    print(f"🎯 Joining room: {ctx.room.name}")
    agent = GeminiVoiceAgentWithRAG(use_rag=True, max_history=10)
    await agent.process_conversation(ctx)


if __name__ == "__main__":
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GOOGLE_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        exit(1)

    print("🚀 Starting Gemini Voice Agent - STRICT RAG MODE (Knowledge Base Only)")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))