# Voice AI System 🎙️

A real-time **Voice AI Platform** enabling natural, multi-turn conversations over **web audio or phone calls**.  
It leverages **React (frontend)**, **Flask + Socket.IO (backend)**, **Whisper (STT)**, **OpenAI + DeepSeek (LLMs)**, and **gTTS (TTS)** to deliver **low-latency, scalable, and intelligent conversations**.  

---

## Key Highlights ✨

- Low latency (<1s) turn handling with streaming audio & caching  
- Full voice pipeline: Voice → Text → AI → Voice  
- Agent-based modular architecture for scalability (1,000+ sessions)  
- Barge-in support → interrupt & re-prompt mid-speech  
- SQLite for persistence + Redis for memory context  
- Analytics Agent for latency/error monitoring  

---

## Screenshots 📸

| Image 1 | Image 2 | Image 3 | Image 4 |
|----------|-------------|-----------------|-----------|
| ![Image 1](https://github.com/ShahzadDiyal/VoiceAISystem/blob/main/frontend/voiceAiSystem/assets/screencapture-1.png) | ![Image 2](https://github.com/ShahzadDiyal/VoiceAISystem/blob/main/frontend/voiceAiSystem/assets/screencapture-2.png) | ![Image 3](https://github.com/ShahzadDiyal/VoiceAISystem/blob/main/frontend/voiceAiSystem/assets/screencapture-3.png) | ![Image 4](https://github.com/ShahzadDiyal/VoiceAISystem/blob/main/frontend/voiceAiSystem/assets/screencapture-4.png) |

---

## System Architecture 🏗️

### High-Level Architecture 🔹

![High-Level Architecture](https://github.com/ShahzadDiyal/VoiceAISystem/blob/main/frontend/voiceAiSystem/assets/High-Level%20Voice%20AI%20Architecture.drawio.png)

- Frontend (React) → Audio capture, transcripts, real-time UI updates  
- Backend (Flask + Socket.IO) → Orchestration, session management  
- Agents → Handle STT, LLM reasoning, TTS, memory, scaling, and analytics  
- External APIs → OpenAI & DeepSeek for LLMs, gTTS for speech synthesis  

---

### Low-Level Architecture 🔹

![Low-Level Architecture](https://github.com/ShahzadDiyal/VoiceAISystem/blob/main/frontend/voiceAiSystem/assets/Low%20level%20diagram.drawio.png)

**Data Flow:**  

1. Session Gateway → Starts client session (100ms setup)  
2. Listener & VAD Agent → Detects speech, applies noise suppression (~50ms)  
3. Speech-to-Text (Whisper) → Generates incremental transcripts (~200ms)  
4. Orchestration Agent → Routes transcripts, merges partials, adds timing (~100ms)  
5. Context & Memory Agent → Summarizes & retrieves context from Redis (~150ms)  
6. LLM Reasoning Agent → Uses OpenAI/DeepSeek to generate structured response (500ms+)  
7. Text-to-Speech Agent → gTTS streams back speech (~300ms)  
8. Analytics Agent → Monitors latency & errors (~50ms)  
9. Coordinator for Scale → Triggers scaling/load shedding (~200ms)  

---

## Technology Stack ⚙️

**Frontend:**  
- React, WebRTC, WebSockets  

**Backend:**  
- Flask + Flask-SocketIO  
- Gunicorn (multi-worker scaling)  

**AI Components:**  
- Speech-to-Text (STT): Whisper (local, fast + private)  
- Language Models (LLMs): OpenAI & DeepSeek APIs (via Azure)  
- Text-to-Speech (TTS): gTTS (streaming synthesis)  

**Data Layer:**  
- SQLite → session persistence  
- Redis → compact context & memory summaries  
- Optional S3/PostgreSQL for storage/logging  

---

## Performance Targets 📊

- First audio reply < 500ms  
- Full conversational turn < 2s  
- Supports 1,000+ concurrent sessions  
- Handles interruptions (barge-in)  

**Latency Budget (per turn):**  
- STT: ~200ms  
- LLM: 500ms+ (API latency bottleneck)  
- TTS: ~300ms  
- Total: ~1s – 1.2s  

---

## Agents Overview 🧩

| Agent | Role | Inputs | Outputs | Latency |
|-------|------|--------|---------|---------|
| Session Gateway | Starts/manages sessions | Client audio | Session ID, audio stream | ~100ms |
| Listener & VAD | Detects voice activity | Audio chunks | Preprocessed audio | ~50ms |
| Speech-to-Text | Transcribes | Audio | Transcripts | ~200ms |
| Orchestration | Routes & merges | Transcripts | Prompts | ~100ms |
| Context & Memory | Stores/retrieves facts | History | Compact context | ~150ms |
| LLM Reasoning | AI responses | Context prompt | Structured response | ~500ms+ |
| Text-to-Speech | Synthesizes voice | Text response | Audio out | ~300ms |
| Analytics | Performance logging | Metrics | Reports | ~50ms |
| Coordinator | Scaling/load balancing | Queue depth | Scaling decisions | ~200ms |

---

## Installation & Setup 🛠️

### Clone Repo  

```bash
git clone https://github.com/yourusername/voice-ai-system.git
cd voice-ai-system
```

---

## Backend Setup (Flask + SocketIO) 🛠 

```bash
cd backend
pip install -r requirements.txt
python app.py
```

## Frontend Setup (React) 🛠 

```bash
cd frontend
npm install
npm start
```





