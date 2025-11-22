#!/bin/bash
echo "Starting LiveKit Token Server..."
cd "$(dirname "$0")"
python token_server.py

