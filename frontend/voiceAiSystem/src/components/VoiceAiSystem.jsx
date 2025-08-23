import React, { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';
import '../../src/App.css';

const BACKEND_URL = 'http://localhost:5000';

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
  const [processingProgress, setProcessingProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAudio, setActiveAudio] = useState(null);
  const [rateLimitError, setRateLimitError] = useState(null);
  const [retryTime, setRetryTime] = useState(0);

  // Refs
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const audioChunksRef = useRef([]);
  const progressIntervalRef = useRef(null);

  // Audio config
  const AUDIO_CONFIG = {
    sampleRate: 16000,
    channels: 1,
    bitsPerSample: 16,
    chunkSize: 1024,
    chunkDuration: 100
  };

  // Initialize WebSocket connection
  useEffect(() => {
    const newSocket = io(BACKEND_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 20000,
      withCredentials: true,
      extraHeaders: {
        "my-custom-header": "abcd"
      }
    });
    
    
    const storedSessionId = localStorage.getItem('session_id');
    if (storedSessionId) {
        newSocket.emit('restore_session', { id: storedSessionId });
    }
    // Connection events
    newSocket.on('connect', () => {
      console.log('Connected to backend');
      setIsConnected(true);
      setSystemStatus('connected');
      setError(null);
      // Explicitly request session creation on connect
      newSocket.emit('create_session');
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend');
      setIsConnected(false);
      setSystemStatus('disconnected');
    });

    newSocket.on('connect_error', (err) => {
      console.error('Connection error:', err);
      setError('Failed to connect to server');
    });

    newSocket.on('reconnect_attempt', () => {
      console.log('Attempting to reconnect...');
    });

    newSocket.on('reconnect_failed', () => {
      console.error('Reconnection failed');
      setError('Connection to server lost. Please refresh the page.');
    });

    // Session handling
    newSocket.on('session_created', (data) => {
      console.log('Session created:', data);
      if (data?.session_id) {
        setSessionId(data.session_id);
        setSystemStatus('connected');
        localStorage.setItem('session_id', data.session_id); // Save session_id
      } else {
        setError('Session creation failed');
      }
    });

    newSocket.on('session_restored', (data) => {
    if (data?.session_id) {
      setSessionId(data.session_id);
      setSystemStatus('connected');
      setConversationHistory(data.history || []);
    }
  });

  
    


    // Transcript handling
    newSocket.on('partial_transcript', (data) => {
      if (data.is_partial) {
        setPartialTranscript(data.text);
      } else {
        setTranscript(data.text);
        setPartialTranscript('');
        setConversationHistory(prev => [
          {
            role: 'user',
            content: data.text,
            timestamp: Date.now(),
            confidence: data.confidence || 0.9
          },
          ...prev
        ]);
      }
    });

    // Audio handling
    newSocket.on('audio_chunk', (data) => {
      if (!isProcessing) {
        setIsProcessing(true);
        setProcessingProgress(0);
        progressIntervalRef.current = setInterval(() => {
          setProcessingProgress(prev => Math.min(prev + 10, 90));
        }, 200);
      }
      
      playAudioChunk(data);
      
      if (data.is_final) {
        setConversationHistory(prev => [
          {
            role: 'assistant',
            content: data.sentence || 'Audio response',
            timestamp: Date.now(),
            type: 'audio'
          },
          ...prev
        ]);
      }
    });

    // Status updates
    newSocket.on('recording_started', () => {
      setSystemStatus('recording');
    });

    newSocket.on('recording_stopped', () => {
      setSystemStatus('processing');
    });

    newSocket.on('processing_complete', () => {
      setProcessingProgress(100);
      setTimeout(() => {
        setIsProcessing(false);
        setProcessingProgress(0);
        clearProgressInterval();
      }, 500);
    });

    // Error handling
    newSocket.on('error', (errorData) => {
      if (errorData.message?.includes('RateLimitReached') || 
          errorData.message?.includes('429')) {
        const waitTime = errorData.details?.match(/wait (\d+) seconds/)?.[1] || 0;
        setRetryTime(parseInt(waitTime));
        setRateLimitError({
          message: 'AI service rate limit reached',
          details: `Please try again in ${Math.ceil(waitTime / 60)} minutes`
        });
      } else {
        setError(errorData.message || 'Unknown error');
      }
    });

    setSocket(newSocket);

  

    return () => {
      newSocket.close();
      clearProgressInterval();

    };
  }, []);

  const clearProgressInterval = () => {
    if (progressIntervalRef.current) {
      clearInterval(progressIntervalRef.current);
      progressIntervalRef.current = null;
    }
  };

  const formatTimeRemaining = (seconds) => {
    if (seconds <= 0) return 'now';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return [
      hours > 0 ? `${hours}h` : '',
      minutes > 0 ? `${minutes}m` : '',
      `${secs}s`
    ].filter(Boolean).join(' ');
  };

  // Audio initialization
  const initializeAudio = useCallback(async () => {
    try {
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
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: AUDIO_CONFIG.sampleRate
      });

      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
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

      return true;
    } catch (error) {
      setError(`Microphone access denied: ${error.message}`);
      return false;
    }
  }, [socket, sessionId]);

  // Audio playback
  const playAudioChunk = useCallback((audioData) => {
    try {
      if (activeAudio) {
        activeAudio.pause();
        URL.revokeObjectURL(activeAudio.src);
      }

      const audioBytes = atob(audioData.audio_data);
      const audioArray = new Uint8Array(audioBytes.length);
      for (let i = 0; i < audioBytes.length; i++) {
        audioArray[i] = audioBytes.charCodeAt(i);
      }

      const audioBlob = new Blob([audioArray], { type: 'audio/mp3' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      const newAudio = new Audio(audioUrl);
      setActiveAudio(newAudio);

      newAudio.onplay = () => setIsPlaying(true);
      newAudio.onended = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        setActiveAudio(null);
      };
      newAudio.onerror = () => {
        setIsPlaying(false);
        URL.revokeObjectURL(audioUrl);
        setActiveAudio(null);
      };

      newAudio.play();
    } catch (error) {
      console.error('Audio playback error:', error);
      setIsProcessing(false);
      clearProgressInterval();
    }
  }, [activeAudio]);

  // Recording controls
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

      if (mediaRecorderRef.current?.state === 'inactive') {
        audioChunksRef.current = [];
        mediaRecorderRef.current.start(AUDIO_CONFIG.chunkDuration);
        setIsRecording(true);
        setError(null);
        socket.emit('start_recording', { session_id: sessionId });
      }
    } catch (error) {
      setError(`Recording failed: ${error.message}`);
    }
  }, [socket, sessionId, initializeAudio]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (socket && sessionId) {
        socket.emit('stop_recording', { session_id: sessionId });
      }
    }
  }, [socket, sessionId]);

  const toggleRecording = useCallback(() => {
    isRecording ? stopRecording() : startRecording();
  }, [isRecording, startRecording, stopRecording]);

  const clearConversation = useCallback(() => {
    setConversationHistory([]);
    setTranscript('');
    setPartialTranscript('');
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      clearProgressInterval();
      if (activeAudio) {
        activeAudio.pause();
        URL.revokeObjectURL(activeAudio.src);
      }
      
    };
  }, [activeAudio]);

  return (
    <div className="voice-ai-app">
      <header className="app-header">
        <h1>🎙️ Voice AI Platform</h1>
        <div className="status-indicators">
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`}></div>
          <span className="status-text">{systemStatus}</span>
          {sessionId && <span className="session-id">Session: {sessionId.slice(0, 8)}</span>}
        </div>
      </header>

      {error && (
        <div className="error-banner">
          <span>⚠️ {error}</span>
          <button onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {rateLimitError && (
        <div className="rate-limit-banner">
          <div className="rate-limit-content">
            <span>⚠️ {rateLimitError.message}</span>
            <span>{rateLimitError.details}</span>
            {retryTime > 0 && (
              <div className="retry-timer">
                <span>Time remaining: {formatTimeRemaining(retryTime)}</span>
                <div className="timer-progress">
                  <div style={{ width: `${100 - (retryTime / (retryTime + 60) * 100)}%` }}></div>
                </div>
              </div>
            )}
          </div>
          <button onClick={() => setRateLimitError(null)}>✕</button>
        </div>
      )}

      <main className="main-content">
        <div className="voice-controls">
          <button
            className={`record-button ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            disabled={!isConnected || !sessionId || isProcessing}
          >
            <div className="record-icon">
              {isRecording ? '⏹️' : '🎤'}
            </div>
            <span>{isRecording ? 'Stop Recording' : 'Start Recording'}</span>
          </button>

          {isProcessing && (
            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${processingProgress}%` }}></div>
              <span className="progress-text">
                {processingProgress < 100 ? `Processing... ${processingProgress}%` : 'Processing complete'}
              </span>
            </div>
          )}

          {isPlaying && (
            <div className="playback-indicator">
              <span>🔊 Playing AI Response...</span>
            </div>
          )}
        </div>

        <div className="transcript-section">
          <h3>Transcript</h3>
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
                Start speaking to see transcription...
              </div>
            )}
          </div>
        </div>

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
                <div key={`${message.timestamp}-${index}`} className={`message ${message.role}`}>
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

      
      </main>

      <footer className="app-footer">
        <div className="footer-info">
          <h3>Built by Shahzad💖</h3>
          <p>Voice AI Platform - Real-time Voice Conversations</p>
          <p>Status: {systemStatus} | Connected: {isConnected ? 'Yes' : 'No'}</p>
        </div>
      </footer>
    </div>
  );
};

export default VoiceAIApp;