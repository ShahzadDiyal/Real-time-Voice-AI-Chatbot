# Voice AI Platform Backend
# Flask application with all required agents using local Whisper for STT and GitHub Models for LLM
# Last updated: August 15, 2025, 4:00 PM PKT

import os
import json
import time
import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from queue import Queue
import logging
import io
import base64
import numpy as np

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import redis
import sqlite3
from concurrent.futures import ThreadPoolExecutor
import pydub
import whisper
import gtts
import soundfile as sf
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "voiceaisystem@15082025"
CORS(app, origins=["http://localhost:5173"])
socketio = SocketIO(
    app,
    cors_allowed_origins="http://localhost:5173",
    async_mode="threading",
    ping_timeout=60,
    ping_interval=25,
)

# Configuration
class Config:
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "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")
    MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "1000"))
    AUDIO_CHUNK_SIZE = 1024
    SAMPLING_RATE = 16000
    LATENCY_TARGET_MS = 500

# Validate configuration
if not Config.GITHUB_TOKEN:
    logger.error("Missing GitHub token")
    raise ValueError("GitHub token is required")

# Data Models
@dataclass
class AudioChunk:
    data: bytes
    timestamp: float
    session_id: str
    chunk_id: str
    is_final: bool = False

@dataclass
class TranscriptData:
    text: str
    confidence: float
    is_partial: bool
    timestamp: float
    session_id: str

@dataclass
class SessionContext:
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    conversation_history: List[Dict]
    context_summary: str
    is_active: bool = True

@dataclass
class LLMResponse:
    text: str
    tool_calls: List[Dict]
    confidence: float
    processing_time: float

# Database setup
def init_db():
    conn = sqlite3.connect("voice_ai.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            created_at REAL,
            last_activity REAL,
            context_summary TEXT,
            is_active BOOLEAN
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            metric_type TEXT,
            value REAL,
            timestamp REAL
        )
        """
    )
    conn.commit()
    conn.close()

# Redis connection
try:
    redis_client = redis.from_url(Config.REDIS_URL)
except Exception as e:
    redis_client = None
    logger.warning(f"Redis not available, using in-memory storage: {e}")

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=20)

# 1. Session Gateway Agent
class SessionGateway:
    def __init__(self):
        self.active_sessions: Dict[str, SessionContext] = {}
        self.session_lock = threading.Lock()

    def create_session(self, user_id: str = None) -> str:
        session_id = str(uuid.uuid4())
        current_time = time.time()
        context = SessionContext(
            session_id=session_id,
            user_id=user_id or f"user_{session_id[:8]}",
            created_at=current_time,
            last_activity=current_time,
            conversation_history=[],
            context_summary="",
        )
        with self.session_lock:
            self.active_sessions[session_id] = context
        self._store_session_db(context)
        logger.info(f"Created session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        with self.session_lock:
            return self.active_sessions.get(session_id)

    def update_session_activity(self, session_id: str):
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].last_activity = time.time()

    def close_session(self, session_id: str):
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].is_active = False
                del self.active_sessions[session_id]
        logger.info(f"Closed session: {session_id}")

    def _store_session_db(self, context: SessionContext):
        conn = sqlite3.connect("voice_ai.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO sessions 
            (id, user_id, created_at, last_activity, context_summary, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                context.session_id,
                context.user_id,
                context.created_at,
                context.last_activity,
                context.context_summary,
                context.is_active,
            ),
        )
        conn.commit()
        conn.close()

# 2. Listener and End-pointing Agent
class ListenerAgent:
    def __init__(self):
        self.voice_activity_threshold = 0.1
        self.silence_timeout = 2.0

    def detect_voice_activity(self, audio_chunk: AudioChunk) -> bool:
        try:
            audio_level = sum(abs(b) for b in audio_chunk.data) / len(audio_chunk.data)
            return audio_level > self.voice_activity_threshold
        except Exception as e:
            logger.error(f"Voice activity detection failed: {e}")
            return False

    def detect_speech_end(self, silence_duration: float) -> bool:
        return silence_duration > self.silence_timeout

    def preprocess_audio(self, audio_data: bytes) -> bytes:
        try:
            return audio_data
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_data

# Audio Chunk Accumulator
class AudioAccumulator:
    def __init__(self):
        self.chunks = {}  # session_id: list of bytes

    def add_chunk(self, session_id: str, audio_data: bytes, is_final: bool):
        if session_id not in self.chunks:
            self.chunks[session_id] = []

        self.chunks[session_id].append(audio_data)

        if is_final:
            full_audio = b"".join(self.chunks[session_id])
            del self.chunks[session_id]
            return full_audio
        return None

audio_accumulator = AudioAccumulator()

# 3. Speech-to-Text Agent (Local Whisper)
class SpeechToTextAgent:
    def __init__(self):
        self.model = whisper.load_model("base")
        self.processing_queue = Queue()
        self.results_cache = {}

    def process_audio_stream(self, full_audio: bytes, session_id: str) -> Optional[TranscriptData]:
        try:
            if not full_audio or len(full_audio) < 100:
                logger.error(f"Invalid or empty audio for session {session_id}")
                socketio.emit(
                    "error",
                    {"message": "Invalid or empty audio data", "session_id": session_id},
                    room=session_id,
                )
                return None

            logger.info(f"Processing full audio for session {session_id}, size: {len(full_audio)} bytes")
            try:
                audio_segment = pydub.AudioSegment.from_file(io.BytesIO(full_audio), format="webm")
            except Exception as e:
                logger.error(f"Failed to parse WebM audio: {e}, attempting conversion")
                audio_segment = pydub.AudioSegment(
                    data=full_audio,
                    sample_width=2,
                    frame_rate=Config.SAMPLING_RATE,
                    channels=1
                )
                if audio_segment is None:
                    logger.error(f"Audio format conversion failed for session {session_id}")
                    socketio.emit(
                        "error",
                        {"message": "Unsupported or corrupted audio format", "session_id": session_id},
                        room=session_id,
                    )
                    return None

            audio_segment = audio_segment.set_channels(1).set_frame_rate(Config.SAMPLING_RATE)
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)

            # Load audio as NumPy array with soundfile
            audio_array, sample_rate = sf.read(wav_io)
            logger.info(f"Loaded audio with sample rate {sample_rate}, dtype {audio_array.dtype}")

            # Ensure dtype is float32 to match Whisper's expectations
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                logger.info("Converted audio array to float32")

            # Verify sample rate matches configuration
            if sample_rate != Config.SAMPLING_RATE:
                logger.warning(f"Resampling audio from {sample_rate} to {Config.SAMPLING_RATE} Hz")
                audio_array = np.interp(
                    np.linspace(0, len(audio_array), int(len(audio_array) * Config.SAMPLING_RATE / sample_rate)),
                    np.arange(len(audio_array)),
                    audio_array
                ).astype(np.float32)

            logger.info(f"Transcribing audio with local Whisper for session {session_id}")
            result = self.model.transcribe(audio_array, fp16=False, language="en")

            transcript_text = result["text"].strip()
            return TranscriptData(
                text=transcript_text,
                confidence=0.9,
                is_partial=False,
                timestamp=time.time(),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Whisper STT error: {e}")
            socketio.emit(
                "error",
                {"message": f"STT error: {str(e)}", "session_id": session_id},
                room=session_id,
            )
            return None

# 4. Orchestration Agent
class OrchestrationAgent:
    def __init__(self, session_gateway, stt_agent, context_agent, llm_agent, tts_agent):
        self.session_gateway = session_gateway
        self.stt_agent = stt_agent
        self.context_agent = context_agent
        self.llm_agent = llm_agent
        self.tts_agent = tts_agent
        self.processing_queues = {}

    def handle_audio_chunk(self, session_id: str, audio_chunk: AudioChunk):
        start_time = time.time()
        try:
            # Convert ArrayBuffer to bytes if necessary
            if isinstance(audio_chunk.data, str):
                audio_chunk.data = base64.b64decode(audio_chunk.data)
            elif not isinstance(audio_chunk.data, bytes):
                logger.error(f"Unexpected audio data type for session {session_id}: {type(audio_chunk.data)}")
                socketio.emit(
                    "error",
                    {"message": "Invalid audio data type", "session_id": session_id},
                    room=session_id,
                )
                return

            full_audio = audio_accumulator.add_chunk(session_id, audio_chunk.data, audio_chunk.is_final)
            if full_audio is not None:
                # Process full audio when is_final = True
                transcript = self.stt_agent.process_audio_stream(full_audio, session_id)
                if not transcript:
                    logger.warning(f"No transcript generated for session {session_id}")
                    return

                self.session_gateway.update_session_activity(session_id)
                self._process_complete_turn(session_id, transcript)

                socketio.emit(
                    "partial_transcript",
                    {
                        "text": transcript.text,
                        "confidence": transcript.confidence,
                        "is_partial": transcript.is_partial,
                        "session_id": session_id,
                    },
                    room=session_id,
                )
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            socketio.emit(
                "error",
                {"message": str(e), "session_id": session_id},
                room=session_id,
            )

    def _process_complete_turn(self, session_id: str, transcript: TranscriptData):
        try:
            context = self.context_agent.get_conversation_context(session_id)
            llm_request = {
                "message": transcript.text,
                "context": context,
                "session_id": session_id,
            }
            llm_response = self.llm_agent.generate_response(llm_request)
            self.context_agent.add_to_conversation(session_id, "user", transcript.text)
            self.context_agent.add_to_conversation(session_id, "assistant", llm_response.text)
            self.tts_agent.synthesize_speech(session_id, llm_response.text)
        except Exception as e:
            logger.error(f"Turn processing error: {e}")
            socketio.emit(
                "error",
                {"message": str(e), "session_id": session_id},
                room=session_id,
            )

# 5. Context and Memory Agent
class ContextMemoryAgent:
    def __init__(self):
        self.conversation_summaries = {}
        self.context_window_size = 10

    def get_conversation_context(self, session_id: str) -> Dict:
        session = session_gateway.get_session(session_id)
        if not session:
            return {}
        recent_history = session.conversation_history[-self.context_window_size:]
        return {
            "recent_history": recent_history,
            "summary": session.context_summary,
            "user_id": session.user_id,
        }

    def add_to_conversation(self, session_id: str, role: str, content: str):
        session = session_gateway.get_session(session_id)
        if session:
            session.conversation_history.append(
                {"role": role, "content": content, "timestamp": time.time()}
            )
            self._store_conversation_db(session_id, role, content)
            if len(session.conversation_history) > 20:
                self._update_summary(session_id)

    def _store_conversation_db(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect("voice_ai.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (session_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, role, content, time.time()),
        )
        conn.commit()
        conn.close()

    def _update_summary(self, session_id: str):
        session = session_gateway.get_session(session_id)
        if session and len(session.conversation_history) > 10:
            recent_topics = [
                msg["content"][:50] + "..." for msg in session.conversation_history[-10:]
            ]
            session.context_summary = "Recent topics: " + "; ".join(recent_topics)

# 6. LLM Reasoning Agent (GitHub Models)
class LLMReasoningAgent:
    def __init__(self):
        self.client = OpenAI(
            base_url="https://models.github.ai/inference/v1",
            api_key=Config.GITHUB_TOKEN,
        )
        self.model_name = "openai/gpt-4o"
        self.response_cache = {}

    def generate_response(self, request: Dict) -> LLMResponse:
        start_time = time.time()
        try:
            cache_key = hash(request["message"])
            if cache_key in self.response_cache:
                cached_response = self.response_cache[cache_key]
                return LLMResponse(
                    text=cached_response,
                    tool_calls=[],
                    confidence=1.0,
                    processing_time=time.time() - start_time,
                )

            messages = self._prepare_messages(request)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=150,
                temperature=0.7,
            )
            response_text = response.choices[0].message.content.strip()

            if len(request["message"].split()) < 5:
                self.response_cache[cache_key] = response_text

            return LLMResponse(
                text=response_text,
                tool_calls=[],
                confidence=0.9,
                processing_time=time.time() - start_time,
            )
        except OpenAIError as e:
            logger.error(f"GitHub Models LLM error: {e}")
            socketio.emit(
                "error",
                {"message": f"LLM error: {str(e)}", "session_id": request["session_id"]},
                room=request["session_id"],
            )
            return LLMResponse(
                text="I apologize, but I'm having trouble processing your request right now.",
                tool_calls=[],
                confidence=0.1,
                processing_time=time.time() - start_time,
            )
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            socketio.emit(
                "error",
                {"message": f"LLM error: {str(e)}", "session_id": request["session_id"]},
                room=request["session_id"],
            )
            return LLMResponse(
                text="I apologize, but I'm having trouble processing your request right now.",
                tool_calls=[],
                confidence=0.1,
                processing_time=time.time() - start_time,
            )

    def _prepare_messages(self, request: Dict) -> List[Dict]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant having a voice conversation. Keep responses concise and natural.",
            }
        ]
        context = request.get("context", {})
        for msg in context.get("recent_history", []):
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": request["message"]})
        return messages

# 7. Text-to-Speech Agent (gTTS)
class TextToSpeechAgent:
    def __init__(self):
        pass

    def synthesize_speech(self, session_id: str, text: str):
        try:
            start_time = time.time()
            sentences = self._chunk_text(text)
            for i, sentence in enumerate(sentences):
                is_final = i == len(sentences) - 1
                logger.info(f"Generating TTS for sentence: {sentence}")
                tts = gtts.gTTS(text=sentence, lang="en", slow=False)
                audio_io = io.BytesIO()
                tts.write_to_fp(audio_io)
                audio_bytes = audio_io.getvalue()
                audio_data = base64.b64encode(audio_bytes).decode()
                socketio.emit(
                    "audio_chunk",
                    {
                        "audio_data": audio_data,
                        "is_final": is_final,
                        "sentence": sentence,
                        "session_id": session_id,
                    },
                    room=session_id,
                )
                time.sleep(0.05)
            processing_time = time.time() - start_time
            logger.info(f"TTS processing time: {processing_time:.2f}s")
        except Exception as e:
            logger.error(f"gTTS TTS error: {e}")
            socketio.emit(
                "error",
                {"message": f"TTS error: {str(e)}", "session_id": session_id},
                room=session_id,
            )

    def _chunk_text(self, text: str) -> List[str]:
        import re
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

# 8. Analytics and Quality Agent
class AnalyticsAgent:
    def __init__(self):
        self.metrics_buffer = []
        self.buffer_size = 100

    def record_metric(self, session_id: str, metric_type: str, value: float):
        metric = {
            "session_id": session_id,
            "metric_type": metric_type,
            "value": value,
            "timestamp": time.time(),
        }
        self.metrics_buffer.append(metric)
        if len(self.metrics_buffer) >= self.buffer_size:
            self._flush_metrics()

    def _flush_metrics(self):
        conn = sqlite3.connect("voice_ai.db")
        cursor = conn.cursor()
        for metric in self.metrics_buffer:
            cursor.execute(
                """
                INSERT INTO analytics (session_id, metric_type, value, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (
                    metric["session_id"],
                    metric["metric_type"],
                    metric["value"],
                    metric["timestamp"],
                ),
            )
        conn.commit()
        conn.close()
        self.metrics_buffer.clear()

    def get_metrics(self, session_id: str = None) -> Dict:
        conn = sqlite3.connect("voice_ai.db")
        cursor = conn.cursor()
        query = """
            SELECT metric_type, AVG(value), COUNT(*) 
            FROM analytics 
        """
        params = []
        if session_id:
            query += "WHERE session_id = ? GROUP BY metric_type"
            params = [session_id]
        else:
            query += "GROUP BY metric_type"
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        metrics = {}
        for metric_type, avg_value, count in results:
            metrics[metric_type] = {"average": avg_value, "count": count}
        return metrics

# 9. Coordinator for Scale
class ScaleCoordinator:
    def __init__(self):
        self.current_load = 0
        self.max_capacity = Config.MAX_CONCURRENT_SESSIONS
        self.queue_depths = {}
        self.latency_metrics = {}

    def check_capacity(self) -> bool:
        return self.current_load < self.max_capacity

    def add_session(self, session_id: str):
        self.current_load += 1
        logger.info(f"Current load: {self.current_load}/{self.max_capacity}")

    def remove_session(self, session_id: str):
        if self.current_load > 0:
            self.current_load -= 1

    def should_load_shed(self) -> bool:
        return self.current_load > self.max_capacity * 0.9

    def get_system_status(self) -> Dict:
        return {
            "current_load": self.current_load,
            "max_capacity": self.max_capacity,
            "utilization": (self.current_load / self.max_capacity) * 100,
            "status": (
                "healthy" if self.current_load < self.max_capacity * 0.8 else "high_load"
            ),
        }

# Initialize all agents
session_gateway = SessionGateway()
listener_agent = ListenerAgent()
stt_agent = SpeechToTextAgent()
context_agent = ContextMemoryAgent()
llm_agent = LLMReasoningAgent()
tts_agent = TextToSpeechAgent()
analytics_agent = AnalyticsAgent()
scale_coordinator = ScaleCoordinator()

orchestration_agent = OrchestrationAgent(
    session_gateway, stt_agent, context_agent, llm_agent, tts_agent
)

# WebSocket Events
@socketio.on("connect")
def handle_connect(auth):
    session_id = session_gateway.create_session()
    join_room(session_id)
    scale_coordinator.add_session(session_id)
    emit("session_created", {"session_id": session_id, "status": "connected"})
    logger.info(f"Client connected, session: {session_id}")

@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id:
        session_gateway.close_session(session_id)
        scale_coordinator.remove_session(session_id)
        leave_room(session_id)
    logger.info(f"Client disconnected: {session_id}")

@socketio.on("audio_chunk")
def handle_audio_chunk(data):
    session_id = data.get("session_id")
    audio_data = data.get("audio_data")
    is_final = data.get("is_final", False)

    if not session_id:
        logger.error("No session_id provided in audio_chunk")
        socketio.emit("error", {"message": "No session_id provided", "session_id": None})
        return

    try:
        # Handle ArrayBuffer or base64 string
        if isinstance(audio_data, str):
            audio_chunk_data = base64.b64decode(audio_data)
        elif isinstance(audio_data, (bytes, bytearray)):
            audio_chunk_data = bytes(audio_data)
        else:
            logger.error(f"Unexpected audio_data type for session {session_id}: {type(audio_data)}")
            socketio.emit(
                "error",
                {"message": "Invalid audio data type", "session_id": session_id},
                room=session_id,
            )
            return
    except Exception as e:
        logger.error(f"Failed to decode audio data for session {session_id}: {e}")
        socketio.emit(
            "error",
            {"message": f"Invalid audio data encoding: {str(e)}", "session_id": session_id},
            room=session_id,
        )
        return

    audio_chunk = AudioChunk(
        data=audio_chunk_data,
        timestamp=time.time(),
        session_id=session_id,
        chunk_id=str(uuid.uuid4()),
        is_final=is_final,
    )
    executor.submit(orchestration_agent.handle_audio_chunk, session_id, audio_chunk)

@socketio.on("start_recording")
def handle_start_recording(data):
    session_id = data.get("session_id")
    if session_id:
        emit("recording_started", {"session_id": session_id}, room=session_id)
        logger.info(f"Recording started for session: {session_id}")

@socketio.on("stop_recording")
def handle_stop_recording(data):
    session_id = data.get("session_id")
    if session_id:
        emit("recording_stopped", {"session_id": session_id}, room=session_id)
        logger.info(f"Recording stopped for session: {session_id}")

# REST API Endpoints
@app.route("/api/health", methods=["GET"])
def health_check():
    system_status = scale_coordinator.get_system_status()
    return jsonify(
        {"status": "healthy", "timestamp": time.time(), "system": system_status}
    )

@app.route("/api/sessions", methods=["POST"])
def create_session_api():
    data = request.json or {}
    user_id = data.get("user_id")
    session_id = session_gateway.create_session(user_id)
    return jsonify({"session_id": session_id, "status": "created"})

@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session_api(session_id):
    session = session_gateway.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(
        {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "is_active": session.is_active,
            "conversation_length": len(session.conversation_history),
        }
    )

@app.route("/api/sessions/<session_id>/history", methods=["GET"])
def get_conversation_history(session_id):
    session = session_gateway.get_session(session_id)
    if not session:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(
        {
            "session_id": session_id,
            "conversation_history": session.conversation_history,
            "context_summary": session.context_summary,
        }
    )

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    session_id = request.args.get("session_id")
    metrics = analytics_agent.get_metrics(session_id)
    return jsonify({"metrics": metrics, "timestamp": time.time()})

@app.route("/api/system/status", methods=["GET"])
def get_system_status():
    return jsonify(scale_coordinator.get_system_status())

# Initialize database
init_db()

if __name__ == "__main__":
    print("Starting Voice AI Platform Backend with local Whisper and GitHub Models...")
    print("WebSocket server running on http://localhost:5000")
    print("REST API available at http://localhost:5000/api/")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)