import React, { useState, useEffect, useRef } from 'react'
import { Room, RoomEvent, RemoteParticipant, Track } from 'livekit-client'
import './VoiceAgent.css'

const VoiceAgent = () => {
  const [room, setRoom] = useState(null)
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [transcript, setTranscript] = useState([])
  const [error, setError] = useState(null)
  const [roomName, setRoomName] = useState('')
  
  const audioRef = useRef(null)
  const roomRef = useRef(null)

  // Generate a random room name if not provided
  useEffect(() => {
    if (!roomName) {
      setRoomName(`room-${Math.random().toString(36).substring(7)}`)
    }
  }, [roomName])

  const getAccessToken = async (roomName) => {
    try {
      const response = await fetch(`/api/token?room=${roomName}`)
      const data = await response.json()
      
      if (!response.ok) {
        const errorMsg = data.error || 'Failed to get access token'
        console.error('Token server error:', errorMsg)
        throw new Error(errorMsg)
      }
      
      if (!data.token) {
        throw new Error('Token server did not return a token')
      }
      
      return data.token
    } catch (err) {
      console.error('Error getting token:', err)
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        throw new Error('Cannot reach token server. Make sure it\'s running on port 8080.')
      }
      throw err
    }
  }

  const testTokenServer = async () => {
    try {
      const response = await fetch('/api/health')
      const data = await response.json()
      if (!data.livekit_configured) {
        throw new Error('Token server: LiveKit credentials not configured')
      }
      return true
    } catch (err) {
      console.error('Token server health check failed:', err)
      throw new Error('Cannot connect to token server. Is it running on port 8080?')
    }
  }

  const connectToRoom = async () => {
    if (isConnecting || isConnected) return

    setIsConnecting(true)
    setError(null)

    try {
      // Test token server first
      await testTokenServer()
      addTranscriptMessage('system', 'Token server connected')
      
      const token = await getAccessToken(roomName)
      const livekitUrl = import.meta.env.VITE_LIVEKIT_URL || 'wss://test-voice-2rh1peeg.livekit.cloud'

      const newRoom = new Room({
        adaptiveStream: true,
        dynacast: true,
      })

      // Set up event handlers
      newRoom.on(RoomEvent.Connected, () => {
        console.log('Connected to room')
        setIsConnected(true)
        setIsConnecting(false)
        addTranscriptMessage('system', 'Connected to room')
      })

      newRoom.on(RoomEvent.Disconnected, () => {
        console.log('Disconnected from room')
        setIsConnected(false)
        addTranscriptMessage('system', 'Disconnected from room')
      })

      newRoom.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
        if (track.kind === 'audio') {
          handleRemoteAudioTrack(track, participant)
          addTranscriptMessage('system', `Agent audio connected`)
        }
      })

      newRoom.on(RoomEvent.TrackPublished, (publication, participant) => {
        if (publication.kind === 'audio' && publication.track) {
          handleRemoteAudioTrack(publication.track, participant)
        }
      })

      newRoom.on(RoomEvent.DataReceived, (payload, participant, kind, topic) => {
        // Handle data channel messages (for transcripts if agent sends them)
        try {
          const data = JSON.parse(new TextDecoder().decode(payload))
          if (data.type === 'transcript') {
            addTranscriptMessage(data.role || 'assistant', data.text)
          }
        } catch (e) {
          // Not JSON or not transcript data
        }
      })

      newRoom.on(RoomEvent.ParticipantConnected, (participant) => {
        if (participant.identity !== newRoom.localParticipant.identity) {
          addTranscriptMessage('system', `${participant.identity} joined`)
        }
      })

      newRoom.on(RoomEvent.ParticipantDisconnected, (participant) => {
        addTranscriptMessage('system', `${participant.identity} left`)
      })

      // Connect to room
      await newRoom.connect(livekitUrl, token)
      
      // Enable microphone
      try {
        await newRoom.localParticipant.setMicrophoneEnabled(true)
        addTranscriptMessage('system', 'Microphone enabled - you can start speaking')
      } catch (err) {
        console.error('Failed to enable microphone:', err)
        addTranscriptMessage('system', 'Warning: Could not enable microphone. Please check permissions.')
      }
      
      setRoom(newRoom)
      roomRef.current = newRoom

    } catch (err) {
      console.error('Connection error:', err)
      const errorMessage = err.message || 'Failed to connect to room'
      setError(errorMessage)
      setIsConnecting(false)
      addTranscriptMessage('system', `Error: ${errorMessage}`)
    }
  }

  const disconnectFromRoom = async () => {
    if (roomRef.current) {
      roomRef.current.disconnect()
      roomRef.current = null
      setRoom(null)
      setIsConnected(false)
      setTranscript([])
    }
  }

  const handleRemoteAudioTrack = (track, participant) => {
    if (track.kind !== 'audio') return

    const audioElement = track.attach()
    audioElement.autoplay = true
    audioRef.current = audioElement
    
    // Add to DOM if not already there
    if (!document.body.contains(audioElement)) {
      audioElement.style.display = 'none'
      document.body.appendChild(audioElement)
    }
  }

  const addTranscriptMessage = (role, text) => {
    setTranscript(prev => [
      ...prev,
      {
        id: Date.now(),
        role,
        text,
        timestamp: new Date().toLocaleTimeString()
      }
    ])
  }

  useEffect(() => {
    return () => {
      if (roomRef.current) {
        roomRef.current.disconnect()
      }
      if (audioRef.current) {
        audioRef.current.remove()
      }
    }
  }, [])

  return (
    <div className="voice-agent">
      <div className="voice-agent-container">
        <div className="voice-agent-header">
          <h1>Voice Agent</h1>
          <p>Talk to the AI assistant powered by knowledge base</p>
        </div>

        <div className="voice-agent-controls">
          {!isConnected ? (
            <div className="connection-panel">
              <div className="room-input-group">
                <label htmlFor="room-name">Room Name:</label>
                <input
                  id="room-name"
                  type="text"
                  value={roomName}
                  onChange={(e) => setRoomName(e.target.value)}
                  placeholder="Enter room name"
                  disabled={isConnecting}
                />
              </div>
              <button
                onClick={connectToRoom}
                disabled={isConnecting || !roomName}
                className="connect-button"
              >
                {isConnecting ? 'Connecting...' : 'Connect'}
              </button>
            </div>
          ) : (
            <div className="connection-panel">
              <div className="status-indicator">
                <span className="status-dot connected"></span>
                <span>Connected to: {roomName}</span>
              </div>
              <button
                onClick={disconnectFromRoom}
                className="disconnect-button"
              >
                Disconnect
              </button>
            </div>
          )}

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        {isConnected && (
          <div className="voice-agent-main">
            <div className="info-panel-full">
              <h2>How to Use</h2>
              <div className="instructions-section">
                <h3>Getting Started</h3>
                <ul>
                  <li>Your microphone is automatically enabled</li>
                  <li>Simply start speaking - the AI agent will listen and respond</li>
                  <li>Agent responses will be played as audio through your speakers</li>
                  <li>You can speak naturally and have a conversation</li>
                </ul>
              </div>

              <div className="instructions-section">
                <h3>What the Agent Knows</h3>
                <ul>
                  <li>The agent answers questions using only the knowledge base</li>
                  <li>If it doesn't know something, it will tell you</li>
                  <li>Keep questions concise for best results</li>
                  <li>You can ask follow-up questions in the same conversation</li>
                </ul>
              </div>

              <div className="audio-status">
                <h3>Connection Status</h3>
                <div className="status-item">
                  <span className="status-label">Microphone:</span>
                  <span className="status-value active">Active</span>
                </div>
                <div className="status-item">
                  <span className="status-label">Speaker:</span>
                  <span className="status-value active">Active</span>
                </div>
                <div className="status-item">
                  <span className="status-label">Room:</span>
                  <span className="status-value">{roomName}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default VoiceAgent

