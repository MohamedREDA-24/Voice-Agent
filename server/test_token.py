"""
Quick test script to verify token generation works
"""
import os
from dotenv import load_dotenv
from pathlib import Path
from livekit.api import AccessToken, VideoGrants

# Load environment
script_dir = Path(__file__).parent
project_root = script_dir.parent
for env_path in [script_dir / ".env", project_root / ".env.local", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    print("ERROR: LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in .env file")
    exit(1)

try:
    token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
        .with_identity("test-user") \
        .with_name("Test User") \
        .with_grants(VideoGrants(
            room_join=True,
            room="test-room",
            can_publish=True,
            can_subscribe=True,
        ))
    
    jwt_token = token.to_jwt()
    print("✅ Token generation successful!")
    print(f"Token length: {len(jwt_token)} characters")
    print(f"Token preview: {jwt_token[:50]}...")
except Exception as e:
    print(f"❌ Token generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

