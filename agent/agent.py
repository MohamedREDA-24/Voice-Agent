# agent.py
import asyncio
import os
import time
from typing import Optional, List
import numpy as np
from dotenv import load_dotenv
from google import genai
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
import pathlib

# Load environment variables
script_dir = pathlib.Path(__file__).parent
project_root = script_dir.parent
for env_path in [script_dir / ".env", project_root / ".env.local", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break


class GeminiVoiceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = "gemini-2.5-flash-native-audio-preview-09-2025"  # More stable for continuous conversation
        
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None
        self._audio_buffer: List[bytes] = []
        self._buffer_threshold = 3200
        self._last_send_time = 0.0
        self._current_session = None
        self._is_running = True

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
        room = ctx.room
        print(f"🔌 Connected to room: {room.name}")

        await self._setup_audio_track(room)

        # Setup audio track handler
        audio_task = None
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.track:
                    print(f"🎤 Found audio track")
                    audio_task = asyncio.create_task(self._handle_audio_track(publication.track))
                    break

        # Run continuous session loop
        try:
            while self._is_running:
                try:
                    print(f"🔄 Starting new Gemini session...")
                    await self._run_single_turn(room)
                    print(f"✅ Turn completed, ready for next question")
                    await asyncio.sleep(0.1)  # Brief pause before next session
                except Exception as e:
                    print(f"⚠️ Session error: {e}, reconnecting...")
                    await asyncio.sleep(1)
        finally:
            self._is_running = False
            if audio_task:
                audio_task.cancel()
            print("🔚 Agent stopped")

    async def _run_single_turn(self, room: rtc.Room):
        """Run a single conversation turn"""
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": "You are a helpful voice assistant. Keep responses brief and natural."
        }
        
        async with self.client.aio.live.connect(model=self.model, config=config) as session:
            self._current_session = session
            
            # Handle responses from this turn
            async for response in session.receive():
                if hasattr(response, "server_content") and response.server_content:
                    server_content = response.server_content
                    
                    if hasattr(server_content, "model_turn") and server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                await self._play_audio(part.inline_data.data, room)
                            
                            if hasattr(part, "text") and part.text:
                                print(f"💬 {part.text}")
                    
                    if hasattr(server_content, "turn_complete") and server_content.turn_complete:
                        print(f"✅ Turn complete")
                        return  # Exit this session
            
            self._current_session = None

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
                pass  # Session might be closing

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

    async def _play_audio(self, audio_data: bytes, room: rtc.Room):
        """Play audio response"""
        try:
            if not self._audio_source:
                return

            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            chunk_size = 480
            
            for i in range(0, len(audio_array), chunk_size):
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
    agent = GeminiVoiceAgent()
    await agent.process_conversation(ctx)


if __name__ == "__main__":
    required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "GOOGLE_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    
    if missing:
        print(f"❌ Missing: {', '.join(missing)}")
        exit(1)

    print("🚀 Starting Gemini Voice Agent")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))