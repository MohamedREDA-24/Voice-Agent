# Quick Start Guide

## Simplest Setup (Recommended)

**Just 1 terminal needed** - Use LiveKit Playground:

### Terminal 1: Voice Agent
```bash
cd agent
python agent.py dev
```

Wait until you see "registered worker" in the logs.

Then:
1. Open https://agents-playground.livekit.io/
2. Join the playground (room auto-created)
3. Start talking!

**That's it!** No token server or web UI needed.

---

## Full Setup (With Custom Web UI)

You need **3 terminals** running simultaneously:

### Terminal 1: Token Server (Required for Web UI)
```bash
cd server
pip install -r requirements.txt
python token_server.py
```

You should see:
```
============================================================
LiveKit Token Server
============================================================
Port: 8080
LiveKit URL: wss://...
API Key: Set
API Secret: Set
============================================================
 * Running on http://0.0.0.0:8080
```

**Keep this terminal open!**

### Terminal 2: Voice Agent
```bash
cd agent
python agent.py dev
```

**Keep this terminal open!**

### Terminal 3: Web UI
```bash
cd web
npm install
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:3000/
```

## Verify Everything is Running

### For Playground Setup (Simplest):
1. **Agent**: Check terminal for "registered worker" message
2. **Playground**: Join room at https://agents-playground.livekit.io/
3. **Agent logs**: Should show "🎯 Joining room" when you join

### For Web UI Setup:
1. **Token Server**: Open http://localhost:8080/api/health in browser
   - Should return: `{"status":"ok","livekit_configured":true,...}`

2. **Web UI**: Open http://localhost:3000
   - Should show the Voice Agent interface

3. **Agent**: Check terminal for "Connected to room" messages

## Common Issues

### Agent Not Connecting
- Make sure agent is running **before** joining playground
- Check `.env` file has correct `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`
- Verify agent shows "registered worker" in logs
- Check agent logs for "🎯 Joining room" when you join playground

### "Cannot connect to token server" (Web UI only)
- Make sure Terminal 1 (token server) is running
- Check it's on port 8080
- Verify no firewall blocking

### "Failed to get access token" (Web UI only)
- Check token server terminal for errors
- Verify `.env` file has `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`
- Run `python server/test_token.py` to test
- Make sure you're using `from livekit.api import AccessToken` (not `livekit_api`)

### Port Already in Use
If port 8080 is busy:
```bash
# Windows: Find what's using the port
netstat -ano | findstr :8080

# Or change port in server/token_server.py or set PORT env var
```

## Setup Options Summary

1. **Simplest**: Agent + Playground (1 terminal)
   - Just run `python agent.py dev`
   - Use https://agents-playground.livekit.io/

2. **Full Control**: Agent + Token Server + Web UI (3 terminals)
   - Custom React UI
   - More control over experience

For more details, see [README.md](README.md) or [ARCHITECTURE.md](ARCHITECTURE.md).

