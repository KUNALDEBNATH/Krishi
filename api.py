"""
Agriculture Chatbot — FastAPI Backend  v10  [Voice-First Edition]
═══════════════════════════════════════════════════════════════════════
Voice input is handled client-side via the Web Speech API.
The browser transcribes speech → text, then POSTs to /api/chat.

Endpoints:
  POST  /api/chat            — voice/text chat with session history
  GET   /api/tts             — text → MP3 audio (gTTS)
  GET   /api/languages       — list supported languages
  GET   /api/history/{sid}   — fetch session conversation history
  DELETE /api/history/{sid}  — clear session history
  GET   /api/status          — server / model health check
  GET   /                    — serve index.html (Siri-style voice UI)

Install:
  pip install fastapi uvicorn gtts deep-translator sentence-transformers
  pip install transformers bitsandbytes accelerate torch

Run:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
═══════════════════════════════════════════════════════════════════════
"""

import io
import time
import uuid
import threading
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import chatbot_core as core

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Agriculture Chatbot API",
    description="Smart RAG ensemble chatbot for agriculture",
    version="10.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    html_file = STATIC_DIR / "index.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return JSONResponse({"status": "API running. Place index.html in ./static/"})


# ─────────────────────────────────────────────────────────────────────────────
# In-memory session store  (auto-expires sessions idle > SESSION_TTL seconds)
# ─────────────────────────────────────────────────────────────────────────────
# Structure: { session_id: { "history": [...], "lang": "en", "last_seen": float } }
_sessions: Dict[str, Dict] = {}
_sessions_lock   = threading.Lock()

MAX_HISTORY_STORED = 20    # max turns kept per session
SESSION_TTL        = 3600  # seconds — expire idle sessions to free memory


def _get_or_create_session(session_id: str) -> Dict:
    with _sessions_lock:
        now = time.time()
        if session_id not in _sessions:
            _sessions[session_id] = {"history": [], "lang": "en", "last_seen": now}
        else:
            _sessions[session_id]["last_seen"] = now
        return _sessions[session_id]


def _append_to_history(session_id: str, role: str, content: str) -> None:
    with _sessions_lock:
        session = _sessions.setdefault(
            session_id, {"history": [], "lang": "en", "last_seen": time.time()}
        )
        session["history"].append({"role": role, "content": content})
        session["last_seen"] = time.time()
        # Trim to max stored turns (keep whole pairs)
        if len(session["history"]) > MAX_HISTORY_STORED * 2:
            session["history"] = session["history"][-(MAX_HISTORY_STORED * 2):]


def _purge_expired_sessions() -> None:
    """Remove sessions idle longer than SESSION_TTL. Called lazily on each chat."""
    cutoff = time.time() - SESSION_TTL
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items() if s.get("last_seen", 0) < cutoff]
        for sid in expired:
            del _sessions[sid]
    if expired:
        print(f"  [Session cleanup] purged {len(expired)} idle session(s)")


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:    str
    session_id: Optional[str]  = None
    language:   Optional[str]  = "en"
    tts:        Optional[bool] = False


class ChatResponse(BaseModel):
    session_id:      str
    answer:          str
    model_used:      Optional[str]
    score:           float
    retrieval_score: float
    language:        str
    candidates:      List[Dict] = []


class LanguageItem(BaseModel):
    code: str
    name: str


# ─────────────────────────────────────────────────────────────────────────────
# POST /api/chat
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Lazy session cleanup (runs fast, no background thread needed)
    _purge_expired_sessions()

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    session    = _get_or_create_session(session_id)

    # Language selection — req.language overrides stored session lang
    lang_code = req.language if req.language in core.LANGUAGES else session["lang"]
    with _sessions_lock:
        session["lang"] = lang_code

    # Auto-detect non-English script in input (overrides explicit selection)
    detected = core.detect_lang(req.message)
    if detected:
        lang_code = detected

    # Translate question to English for retrieval + scoring
    query_en = req.message.strip()
    if lang_code != "en":
        translated_q = core.translate_question_to_english(req.message)
        if translated_q and translated_q.strip() and translated_q != req.message:
            query_en = translated_q.strip()

    history = session["history"].copy()

    result = core.predict(
        user_input  = query_en,
        lang_code   = lang_code,
        history     = history,
        tts_enabled = req.tts or False,
        verbose     = True,
    )

    # Persist turns (store English question + final answer)
    _append_to_history(session_id, "user",      query_en)
    _append_to_history(session_id, "assistant", result["answer"])

    return ChatResponse(
        session_id      = session_id,
        answer          = result["answer"],
        model_used      = result.get("model_used"),
        score           = result.get("score", 0.0),
        retrieval_score = result.get("retrieval_score", 0.0),
        language        = lang_code,
        candidates      = result.get("candidates", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/tts
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/tts")
async def text_to_speech(
    text:     str           = Query(..., description="Text to convert to speech"),
    language: Optional[str] = Query(default="en", description="Language code"),
):
    """Convert text to MP3 audio and stream it back."""
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    # Clamp to 3 000 chars to avoid runaway TTS calls
    text = text[:3000]

    lang_info = core.LANGUAGES.get(language, core.LANGUAGES["en"])
    gtts_lang = lang_info["gtts_lang"]

    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=reply.mp3"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/languages
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/languages", response_model=List[LanguageItem])
async def get_languages():
    return [
        LanguageItem(code=code, name=info["name"])
        for code, info in core.LANGUAGES.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/history/{session_id}
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    with _sessions_lock:
        session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "language":   session["lang"],
        "turns":      len(session["history"]) // 2,
        "history":    session["history"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /api/history/{session_id}
# ─────────────────────────────────────────────────────────────────────────────
@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id]["history"] = []
    return {"status": "cleared", "session_id": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# GET /api/status
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    import torch
    return {
        "status":     "running",
        "device":     core.DEVICE,
        "generators": [g["meta"]["label"] for g in core.generators.values()],
        "kb_size":    len(core.questions),
        "sessions":   len(_sessions),
        "gpu":        torch.cuda.get_device_name(0) if core.DEVICE == "cuda" else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
