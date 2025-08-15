// Voice AI Platform Frontend - React Application
// Main App Component with complete voice interaction features

import React, { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';
import '../../src/App.css';

// WebSocket connection
const BACKEND_URL = 'http://localhost:5000';

// Main Voice AI Application Component
const VoiceAIApp = () => {
  // State management
  const [socket, setSocket] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [partialTranscript, setPartialTranscript] = useState('');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [systemStatus, setSystemStatus] = useState('disconnected');
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState({});

  // Refs for media handling
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Audio configuration
  const AUDIO_CONFIG = {
    sampleRate: 16000,
    channels: 1,
    bitsPerSample: 16,
    chunkSize: 1024,
    chunkDuration: 100 // milliseconds
  };

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io(BACKEND_URL, {
      transports: ['websocket'],
      timeout: 20000,
    });

    newSocket.on('connect', () => {
      console.log('Connected to Voice AI backend');
      setIsConnected(true);
      setSystemStatus('connected');
      setError(null);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setIsConnected(false);
      setSystemStatus('disconnected');
    });

    newSocket.on('session_created', (data) => {
      console.log('Session created:', data);
      if (data.session_id) {
        setSessionId(data.session_id);
        setSystemStatus('ready');
      } else {
        setError('Session creation failed: No session ID received');
      }
    });

    newSocket.on('partial_transcript', (data) => {
      console.log('Partial transcript:', data);
      if (data.is_partial) {
        setPartialTranscript(data.text);
      } else {
        setTranscript(data.text);
        setPartialTranscript('');
        
        // Add user message to conversation
        setConversationHistory(prev => [...prev, {
          role: 'user',
          content: data.text,
          timestamp: Date.now(),
          confidence: data.confidence || 0.9
        }]);
      }
    });

    newSocket.on('audio_chunk', (data) => {
      console.log('Received audio chunk:', data);
      playAudioChunk(data);
      
      // Add AI response to conversation when complete
      if (data.is_final) {
        setConversationHistory(prev => [...prev, {
          role: 'assistant',
          content: data.sentence || 'Audio response',
          timestamp: Date.now(),
          type: 'audio'
        }]);
      }
    });

    newSocket.on('recording_started', () => {
      setSystemStatus('recording');
    });

    newSocket.on('recording_stopped', () => {
      setSystemStatus('processing');
    });

    newSocket.on('error', (errorData) => {
      console.error('Backend error:', errorData);
      setError(errorData.message || 'Unknown error');
    });

    setSocket(newSocket);

    return () => {
      if (newSocket) {
        newSocket.close();
      }
    };
  }, []);

  // Initialize audio context and media stream
  const initializeAudio = useCallback(async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: AUDIO_CONFIG.sampleRate,
          channelCount: AUDIO_CONFIG.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      streamRef.current = stream;

      // Initialize AudioContext for processing
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: AUDIO_CONFIG.sampleRate
      });

      // Initialize MediaRecorder for capturing audio
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          
          // Send audio chunk to backend as base64
          if (socket && sessionId) {
            const reader = new FileReader();
            reader.onloadend = () => {
              const arrayBuffer = reader.result;
              const uint8Array = new Uint8Array(arrayBuffer);
              const base64String = btoa(String.fromCharCode(...uint8Array));
              socket.emit('audio_chunk', {
                session_id: sessionId,
                audio_data: base64String,
                is_final: false,
                timestamp: Date.now()
              });
            };
            reader.readAsArrayBuffer(event.data);
          }
        }
      };

      mediaRecorderRef.current.onstop = () => {
        // Send final audio chunk
        if (socket && sessionId && audioChunksRef.current.length > 0) {
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const reader = new FileReader();
          reader.onloadend = () => {
            const arrayBuffer = reader.result;
            const uint8Array = new Uint8Array(arrayBuffer);
            const base64String = btoa(String.fromCharCode(...uint8Array));
            socket.emit('audio_chunk', {
              session_id: sessionId,
              audio_data: base64String,
              is_final: true,
              timestamp: Date.now()
            });
          };
          reader.readAsArrayBuffer(audioBlob);
        }
        audioChunksRef.current = [];
      };

      console.log('Audio initialization successful');
      return true;
    } catch (error) {
      console.error('Audio initialization failed:', error);
      setError(`Microphone access denied: ${error.message}`);
      return false;
    }
  }, [socket, sessionId]);

  // Start recording audio
  const startRecording = useCallback(async () => {
    if (!socket || !sessionId) {
      setError('No active session');
      return;
    }

    try {
      if (!streamRef.current) {
        const success = await initializeAudio();
        if (!success) return;
      }

      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'inactive') {
        audioChunksRef.current = [];
        mediaRecorderRef.current.start(AUDIO_CONFIG.chunkDuration);
        setIsRecording(true);
        setError(null);
        
        socket.emit('start_recording', { session_id: sessionId });
        console.log('Recording started');
      }
    } catch (error) {
      console.error('Failed to start recording:', error);
      setError(`Recording failed: ${error.message}`);
    }
  }, [socket, sessionId, initializeAudio]);

  // Stop recording audio
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (socket && sessionId) {
        socket.emit('stop_recording', { session_id: sessionId });
      }
      console.log('Recording stopped');
    }
  }, [socket, sessionId]);

  // Play audio chunk received from backend
  const playAudioChunk = useCallback((audioData) => {
    try {
      // Decode base64 audio data
      const audioBytes = atob(audioData.audio_data);
      const audioArray = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i);
      }

      // Create audio blob and play
      const audioBlob = new Blob([audioArray], { type: 'audio/mp3' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const audio = new Audio(audioUrl);
      audio.onloadeddata = () => {
        setIsPlaying(true);
        audio.play().catch(error => {
          console.error('Audio playback failed:', error);
        });
      };
      
      audio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

      audio.onerror = (error) => {
        console.error('Audio playback error:', error);
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
      };

    } catch (error) {
      console.error('Audio processing error:', error);
    }
  }, []);

  // Toggle recording state
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // Clear conversation history
  const clearConversation = useCallback(() => {
    setConversationHistory([]);
    setTranscript('');
    setPartialTranscript('');
  }, []);

  // Fetch system metrics
  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/analytics${sessionId ? `?session_id=${sessionId}` : ''}`);
      const data = await response.json();
      setMetrics(data.metrics || {});
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  }, [sessionId]);

  // Auto-fetch metrics periodically
  useEffect(() => {
    if (isConnected) {
      fetchMetrics();
      const interval = setInterval(fetchMetrics, 5000);
      return () => clearInterval(interval);
    }
  }, [isConnected, fetchMetrics]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
    };
  }, []);

  // Render component
  return (
    <div className="voice-ai-app">
      {/* Header */}
      <header className="app-header">
        <h1>🎙️ Voice AI Platform</h1>
        <div className="status-indicators">
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></div>
          <span className="status-text">{systemStatus}</span>
          {sessionId && <span className="session-id">Session: {sessionId.slice(0, 8)}</span>}
        </div>
      </header>

      {/* Error Display */}
      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Main Interface */}
      <main className="main-content">
        {/* Voice Controls */}
        <div className="voice-controls">
          <button
            className={`record-button ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            disabled={!isConnected || !sessionId}
          >
            <div className="record-icon">
              {isRecording ? '⏹️' : '🎤'}
            </div>
            <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
          </button>

          {isPlaying && (
            <div className="playback-indicator">
              <span>🔊 Playing AI Response...</span>
            </div>
          )}
        </div>

        {/* Live Transcript */}
        <div className="transcript-section">
          <h3>Live Transcript</h3>
          <div className="transcript-box">
            {partialTranscript && (
              <div className="partial-transcript">
                <em>{partialTranscript}</em>
              </div>
            )}
            {transcript && (
              <div className="final-transcript">
                {transcript}
              </div>
            )}
            {!transcript && !partialTranscript && (
              <div className="transcript-placeholder">
                Start speaking to see live transcription...
              </div>
            )}
          </div>
        </div>

        {/* Conversation History */}
        <div className="conversation-section">
          <div className="conversation-header">
            <h3>Conversation History</h3>
            <button className="clear-button" onClick={clearConversation}>
              Clear
            </button>
          </div>
          <div className="conversation-history">
            {conversationHistory.length === 0 ? (
              <div className="conversation-placeholder">
                No conversation yet. Start by recording your voice!
              </div>
            ) : (
              conversationHistory.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-header">
                    <span className="role-badge">
                      {message.role === 'user' ? '👤' : '🤖'} {message.role}
                    </span>
                    <span className="timestamp">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </span>
                    {message.confidence && (
                      <span className="confidence">
                        {Math.round(message.confidence * 100)}% confidence
                      </span>
                    )}
                  </div>
                  <div className="message-content">
                    {message.content}
                    {message.type === 'audio' && <span className="audio-indicator">🔊</span>}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* System Metrics */}
        <div className="metrics-section">
          <h3>System Metrics</h3>
          <div className="metrics-grid">
            {Object.entries(metrics).map(([key, value]) => (
              <div key={key} className="metric-card">
                <div className="metric-label">{key.replace(/_/g, ' ')}</div>
                <div className="metric-value">
                  {typeof value === 'object' ? 
                    `${value.average?.toFixed(2)} (${value.count} samples)` : 
                    value
                  }
                </div>
              </div>
            ))}
            {Object.keys(metrics).length === 0 && (
              <div className="no-metrics">No metrics available yet</div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-info">
          <p>Voice AI Platform - Real-time Voice Conversations</p>
          <p>Status: {systemStatus} | Connected: {isConnected ? 'Yes' : 'No'}</p>
        </div>
      </footer>
    </div>
  );
};

export default VoiceAIApp;