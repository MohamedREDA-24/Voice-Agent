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
    def __init__(self, use_rag: bool = True):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash-native-audio-preview-09-2025"
        self.use_rag = use_rag
        
        # Initialize RAG system
        self.rag = None
        self.knowledge_base_summary = ""
        
        if self.use_rag:
            print("=" * 60)
            print("🔧 INITIALIZING RAG SYSTEM")
            print("=" * 60)
            try:
                self.rag = RAGSystem(embedding_model="google")
                print("✅ RAG system initialized")
                
                # Build/load index
                if self.rag.index is None or len(self.rag.metadata) == 0:
                    print("📚 Building RAG index...")
                    self.rag.build_index()
                    
                if self.rag.index is not None and len(self.rag.metadata) > 0:
                    print(f"✅ RAG ready with {len(self.rag.metadata)} chunks")
                    
                    # Prepare knowledge base summary for system prompt
                    self.knowledge_base_summary = self._prepare_knowledge_summary()
                    print(f"✅ Prepared knowledge base summary ({len(self.knowledge_base_summary)} chars)")
                else:
                    print("⚠️ WARNING: No documents indexed")
                    self.use_rag = False
                    
            except Exception as e:
                print(f"❌ Failed to initialize RAG: {e}")
                import traceback
                traceback.print_exc()
                self.rag = None
                self.use_rag = False
            
            print("=" * 60)
        
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None
        self._audio_buffer: List[bytes] = []
        self._buffer_threshold = 3200
        self._last_send_time = 0.0
        self._current_session = None
        self._is_running = True
        self._room = None

    def _prepare_knowledge_summary(self) -> str:
        """Prepare a comprehensive knowledge base summary from all chunks"""
        if not self.rag or not self.rag.metadata:
            return ""
        
        # Get all unique content from the knowledge base
        all_content = []
        seen_content = set()
        
        for meta in self.rag.metadata:
            content = meta.get('content', '').strip()
            # Use first 100 chars as key to avoid exact duplicates
            content_key = content[:100]
            if content and content_key not in seen_content:
                seen_content.add(content_key)
                all_content.append(content)
        
        # Combine into a structured knowledge base
        knowledge_text = "\n\n".join(all_content[:20])  # Limit to 20 chunks to stay within limits
        
        return f"""KNOWLEDGE BASE:
{knowledge_text}

You must answer questions using ONLY the information from this knowledge base. If the answer is not in the knowledge base, say "I don't have that information in my knowledge base."
"""

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

        # Start session
        session_task = asyncio.create_task(self._maintain_session())
        
        # Setup audio track handler
        audio_task = None
        
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
        """Run continuous audio session with knowledge base in system prompt"""
        
        # Build system instruction with embedded knowledge base
        base_instruction = """You are a helpful voice assistant with access to a knowledge base. 

RULES:
1. Answer questions using the knowledge base provided below
2. Be conversational and natural - you're speaking, not writing
3. Keep responses concise (2-3 sentences max)
4. If the answer isn't in the knowledge base, say "I don't have that information"
5. Never make up information not in the knowledge base

"""
        
        system_instruction = base_instruction + self.knowledge_base_summary if self.use_rag else base_instruction

        config = {
            "response_modalities": ["AUDIO"],  # Audio-only for native audio model
            "system_instruction": system_instruction,
            "speech_config": {
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Puck"
                    }
                }
            }
        }
        
        rag_status = "ENABLED ✅" if self.use_rag else "DISABLED ❌"
        print(f"\n🔄 Starting session (RAG: {rag_status})")
        print(f"   System prompt length: {len(system_instruction)} chars")
        
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            self._current_session = session
            print(f"✅ Session connected with knowledge base embedded\n")
            
            # Main loop
            while self._is_running:
                try:
                    async for response in session.receive():
                        if not self._is_running:
                            break
                        await self._handle_response(response)
                except Exception as e:
                    print(f"⚠️ Response error: {e}")
                    if self._is_running:
                        await asyncio.sleep(0.1)
                    else:
                        break

    async def _handle_response(self, response):
        """Handle audio responses"""
        if not hasattr(response, "server_content") or not response.server_content:
            return
        
        server_content = response.server_content
        
        # Handle model's audio response
        if hasattr(server_content, "model_turn") and server_content.model_turn:
            for part in server_content.model_turn.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    await self._play_audio(part.inline_data.data)
                
                if hasattr(part, "text") and part.text:
                    print(f"🤖 {part.text}")
        
        # Turn complete
        if hasattr(server_content, "turn_complete") and server_content.turn_complete:
            print("✅ Turn complete\n")

    async def _handle_audio_track(self, track: rtc.Track):
        """Process incoming audio"""
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
            print("🎧 Audio stream cancelled")
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
    agent = GeminiVoiceAgentWithRAG(use_rag=True)
    await agent.process_conversation(ctx)


if __name__ == "__main__":
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GOOGLE_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        exit(1)

    print("=" * 60)
    print("🚀 Starting Gemini Voice Agent with RAG")
    print("=" * 60)
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))