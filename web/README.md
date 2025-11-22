# Voice Agent Web UI

React frontend for the Voice Agent with LiveKit integration.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create a `.env` file with your LiveKit URL:
```
VITE_LIVEKIT_URL=wss://your-livekit-server.com
```

3. Start the development server:
```bash
npm run dev
```

4. Make sure the token server is running (see `server/token_server.py`)

## Features

- Connect to LiveKit rooms
- Real-time audio communication with the agent
- Live transcript display
- Clean, modern UI

## Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

