@echo off
echo Starting LiveKit Token Server...
echo.
cd /d %~dp0
python token_server.py
pause

