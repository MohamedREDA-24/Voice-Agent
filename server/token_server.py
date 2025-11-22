"""
LiveKit Token Server
Generates access tokens for the web UI to connect to LiveKit rooms.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from livekit.api import AccessToken, VideoGrants
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
script_dir = Path(__file__).parent
project_root = script_dir.parent
for env_path in [script_dir / ".env", project_root / ".env.local", project_root / ".env"]:
    if env_path.exists():
        load_dotenv(env_path)
        break

app = Flask(__name__)
CORS(app)  # Enable CORS for the React frontend

# Get LiveKit credentials
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://test-voice-2rh1peeg.livekit.cloud")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")


@app.route('/api/token', methods=['GET'])
def get_token():
    """Generate a LiveKit access token for the given room"""
    room_name = request.args.get('room', 'default-room')
    participant_name = request.args.get('name', 'user')
    
    # Check credentials
    if not LIVEKIT_API_KEY:
        print("ERROR: LIVEKIT_API_KEY not set")
        return jsonify({'error': 'LIVEKIT_API_KEY not configured'}), 500
    
    if not LIVEKIT_API_SECRET:
        print("ERROR: LIVEKIT_API_SECRET not set")
        return jsonify({'error': 'LIVEKIT_API_SECRET not configured'}), 500
    
    try:
        print(f"Generating token for room: {room_name}, participant: {participant_name}")
        
        # Create access token
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(participant_name) \
            .with_name(participant_name) \
            .with_grants(VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            ))
        
        jwt_token = token.to_jwt()
        
        print(f"Token generated successfully for room: {room_name}")
        
        return jsonify({
            'token': jwt_token,
            'url': LIVEKIT_URL,
            'room': room_name
        })
    
    except ImportError as e:
        error_msg = f"Import error: {str(e)}. Make sure livekit-api is installed: pip install livekit-api==1.0.7"
        print(f"ERROR: {error_msg}")
        return jsonify({'error': error_msg}), 500
    except Exception as e:
        error_msg = f"Token generation failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'livekit_configured': bool(LIVEKIT_API_KEY and LIVEKIT_API_SECRET),
        'livekit_url': LIVEKIT_URL
    })


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print("=" * 60)
    print("LiveKit Token Server")
    print("=" * 60)
    print(f"Port: {port}")
    print(f"LiveKit URL: {LIVEKIT_URL}")
    print(f"API Key: {'Set' if LIVEKIT_API_KEY else 'NOT SET'}")
    print(f"API Secret: {'Set' if LIVEKIT_API_SECRET else 'NOT SET'}")
    print("=" * 60)
    
    if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
        print("WARNING: LiveKit credentials not configured!")
        print("Please set LIVEKIT_API_KEY and LIVEKIT_API_SECRET in your .env file")
    
    app.run(host='0.0.0.0', port=port, debug=True)
