import asyncio
import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from livekit import agents, rtc
from livekit.agents import AgentServer, JobContext, WorkerOptions
from livekit.rtc import TrackPublishedEvent, TrackKind

load_dotenv(".env.local")

class GeminiVoiceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self._audio_source = None
        self._audio_track = None
        self._track_published = False
    
    async def process_conversation(self, ctx: JobContext):
        """Main method to handle the voice conversation"""
        room = ctx.room
        
        try:
            async with self.client.aio.live.connect(
                model="gemini-2.0-flash-exp",  # Use correct model name
                config={"response_modalities": ["AUDIO"]}
            ) as live_session:
                
                print("🔗 Connected to Gemini Live API")
                
                # Start processing Gemini responses
                response_task = asyncio.create_task(
                    self._handle_gemini_responses(live_session, room)
                )
                
                # Subscribe to participant audio and forward to Gemini
                audio_task = asyncio.create_task(
                    self._handle_participant_audio(live_session, room)
                )
                
                # Wait for either task to complete or fail
                done, pending = await asyncio.wait(
                    [response_task, audio_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
        except Exception as e:
            print(f"❌ Error in Gemini conversation: {e}")
            import traceback
            traceback.print_exc()
            try:
                await room.local_participant.publish_data(
                    f"Error: {str(e)}".encode(),
                    topic="error"
                )
            except:
                pass
        finally:
            # Cleanup audio resources
            await self._cleanup_audio()

    async def _handle_participant_audio(self, live_session, room):
        """Handle incoming audio from participants and forward to Gemini"""
        async for event in room:
            if isinstance(event, TrackPublishedEvent):
                track = event.track
                if (track.kind == TrackKind.KIND_AUDIO and 
                    event.participant != room.local_participant):
                    print(f"🎤 Subscribed to audio track from {event.participant.identity}")
                    
                    async for frame in track:
                        # Convert audio frame to bytes (PCM format)
                        pcm_data = self._audio_frame_to_pcm(frame)
                        
                        if pcm_data:
                            try:
                                await live_session.send_realtime_input(
                                    audio=genai.types.Blob(
                                        data=pcm_data,
                                        mime_type='audio/pcm;rate=16000;channels=1'
                                    )
                                )
                            except Exception as e:
                                print(f"⚠️ Error sending audio to Gemini: {e}")
                                break

    def _audio_frame_to_pcm(self, frame):
        """Convert LiveKit audio frame to PCM bytes"""
        try:
            # Convert the frame data to bytes
            # The exact conversion depends on your audio format
            if hasattr(frame, 'data'):
                return bytes(frame.data)
            else:
                return frame.bytes_data  # Alternative attribute name
        except Exception as e:
            print(f"Error converting audio frame: {e}")
            return None

    async def _handle_gemini_responses(self, live_session, room):
        """Handle responses from Gemini and forward to room"""
        try:
            async for response in live_session.receive():
                print(f"📨 Received response from Gemini: {response}")
                
                if response.audio and response.audio.data:
                    print("🔊 Playing Gemini audio response")
                    # Convert Gemini audio to LiveKit format and play
                    await self._play_audio_response(response.audio.data, room)
                
                if response.text:
                    print(f"💬 Gemini text: {response.text}")
                    # You can also display the text in the UI
                    await room.local_participant.publish_data(
                        response.text.encode(),
                        topic="transcript"
                    )
                    
        except Exception as e:
            print(f"❌ Error handling Gemini responses: {e}")

    async def _play_audio_response(self, audio_data, room):
        """Play audio response in the room"""
        try:
            # Create an audio track from the Gemini audio data
            # This is a simplified version - you might need to adjust based on audio format
            from livekit import rtc
            
            # Create audio source
            audio_source = rtc.AudioSource(
                sample_rate=24000,  # Gemini's output sample rate
                channels=1
            )
            
            # Create track from source
            audio_track = rtc.LocalAudioTrack.create_audio_track(
                "gemini_audio", 
                audio_source
            )
            
            # Publish the track
            await room.local_participant.publish_track(audio_track)
            
            # Convert and push audio data (simplified - you'd need proper chunking)
            # This part depends on the exact audio format from Gemini
            
        except Exception as e:
            print(f"❌ Error playing audio response: {e}")

# Create and configure the agent server
server = AgentServer(
    worker_options=WorkerOptions(
        name="gemini-voice-agent"
    )
)

@server.rtc_session()
async def _handle_rtc_session(ctx: JobContext):
    """Handle new LiveKit WebRTC sessions"""
    room = ctx.room
    print(f"🎯 Agent joining room: {room.name}")
    
    agent = GeminiVoiceAgent()
    await agent.process_conversation(ctx)

if __name__ == "__main__":
    # Check environment variables
    required_vars = ['LIVEKIT_API_KEY', 'LIVEKIT_API_SECRET', 'GOOGLE_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            print(f"❌ Missing required environment variable: {var}")
            exit(1)
    
    print("🚀 Starting Gemini Voice Agent...")
    agents.cli.run_app(server)