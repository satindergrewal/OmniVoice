"""OpenAI-compatible TTS proxy for OmniVoice.

Exposes the OpenAI `POST /v1/audio/speech` endpoint backed by a locally
loaded OmniVoice model, so any client that speaks the OpenAI TTS API can
drive OmniVoice with no client-side changes.

Run:
    pip install -r openai_proxy/requirements.txt
    python -m openai_proxy.app            # or: uvicorn openai_proxy.app:app

Environment variables:
    OMNIVOICE_MODEL     HF repo id or local path   (default: k2-fsa/OmniVoice)
    OMNIVOICE_DEVICE    cuda / cpu / mps / xpu     (default: auto-detected)
    OMNIVOICE_DTYPE     float16 / bfloat16 / float32 (default: float16)
    OMNIVOICE_VOICES    path to voices.json        (default: ./voices.json)
    OMNIVOICE_API_KEY   if set, require "Authorization: Bearer <key>"
    OMNIVOICE_HOST      bind host                  (default: 0.0.0.0)
    OMNIVOICE_PORT      bind port                  (default: 8000)
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from omnivoice.models.omnivoice import OmniVoice
from omnivoice.utils.common import get_best_device

logger = logging.getLogger("omnivoice.openai_proxy")
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    level=logging.INFO,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MODEL_ID = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("OMNIVOICE_DEVICE") or get_best_device()
DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}.get(os.environ.get("OMNIVOICE_DTYPE", "float16"), torch.float16)
VOICES_PATH = Path(
    os.environ.get("OMNIVOICE_VOICES", Path(__file__).parent / "voices.json")
)
API_KEY = os.environ.get("OMNIVOICE_API_KEY")

# Generation defaults (mirror omnivoice-infer CLI).
GEN_DEFAULTS = dict(
    num_step=32,
    guidance_scale=2.0,
    t_shift=0.1,
    denoise=True,
    postprocess_output=True,
    layer_penalty_factor=5.0,
    position_temperature=5.0,
    class_temperature=0.0,
)

# OpenAI response_format -> (soundfile subtype | "pydub:<fmt>" | "pcm", content-type)
FORMAT_MAP = {
    "wav": ("sf:WAV", "audio/wav"),
    "flac": ("sf:FLAC", "audio/flac"),
    "mp3": ("pydub:mp3", "audio/mpeg"),
    "opus": ("pydub:opus", "audio/ogg"),
    "aac": ("pydub:adts", "audio/aac"),
    "pcm": ("pcm", "audio/pcm"),
}

# --------------------------------------------------------------------------- #
# Voice registry
# --------------------------------------------------------------------------- #


def load_voices() -> dict:
    if not VOICES_PATH.exists():
        logger.warning("voices file %s not found; using built-in defaults", VOICES_PATH)
        return {
            "default_voice": "alloy",
            "voices": {"alloy": {"instruct": "female, young adult, moderate pitch"}},
        }
    with open(VOICES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


VOICES = load_voices()


def resolve_voice(name: Optional[str]) -> dict:
    """Map an OpenAI voice name to an OmniVoice voice spec dict."""
    voices = VOICES.get("voices", {})
    # Keys starting with "_" are treated as comments/templates: never selectable.
    if name and not name.startswith("_") and name in voices:
        return voices[name]
    default = VOICES.get("default_voice")
    if default and default in voices:
        logger.info("voice %r unknown, falling back to default %r", name, default)
        return voices[default]
    # Last resort: treat the requested name as a literal instruct string.
    return {"instruct": "female, young adult, moderate pitch"}


# --------------------------------------------------------------------------- #
# Model (loaded once at startup, calls serialized)
# --------------------------------------------------------------------------- #

MODEL: Optional[OmniVoice] = None
_GEN_LOCK = asyncio.Lock()


def load_model() -> OmniVoice:
    logger.info("Loading OmniVoice %s on %s (%s) ...", MODEL_ID, DEVICE, DTYPE)
    model = OmniVoice.from_pretrained(MODEL_ID, device_map=DEVICE, dtype=DTYPE)
    model.eval()
    logger.info("Model ready. sampling_rate=%s", model.sampling_rate)
    return model


# --------------------------------------------------------------------------- #
# Audio encoding
# --------------------------------------------------------------------------- #


def encode_audio(wav: np.ndarray, sr: int, response_format: str) -> bytes:
    """Encode a float32 mono waveform [-1, 1] to the requested format."""
    fmt = response_format.lower()
    if fmt not in FORMAT_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{response_format}'. "
            f"Supported: {', '.join(FORMAT_MAP)}.",
        )
    kind, _ = FORMAT_MAP[fmt]
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    if kind == "pcm":
        # OpenAI pcm = signed 16-bit little-endian mono, raw (no header).
        pcm16 = np.clip(wav, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype("<i2")
        return pcm16.tobytes()

    if kind.startswith("sf:"):
        buf = io.BytesIO()
        sf.write(buf, wav, sr, format="WAV" if "WAV" in kind else "FLAC")
        return buf.getvalue()

    if kind.startswith("pydub:"):
        from pydub import AudioSegment

        target = kind.split(":", 1)[1]
        pcm16 = (np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2")
        seg = AudioSegment(
            data=pcm16.tobytes(), sample_width=2, frame_rate=sr, channels=1
        )
        out = io.BytesIO()
        try:
            seg.export(out, format=target)
        except Exception as exc:  # ffmpeg missing or codec unavailable
            raise HTTPException(
                status_code=500,
                detail=f"Failed to encode '{fmt}' (is ffmpeg installed?): {exc}",
            )
        return out.getvalue()

    raise HTTPException(status_code=500, detail="encoder misconfigured")


# --------------------------------------------------------------------------- #
# Request schema (OpenAI-compatible + OmniVoice extras)
# --------------------------------------------------------------------------- #


class SpeechRequest(BaseModel):
    model: str = "omnivoice"
    input: str
    voice: Optional[str] = None
    response_format: str = "mp3"
    speed: float = 1.0
    # OmniVoice extras (ignored by stock OpenAI clients; usable via extra_body)
    language: Optional[str] = None
    instruct: Optional[str] = None
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None

    class Config:
        extra = "ignore"


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #

app = FastAPI(title="OmniVoice OpenAI-compatible TTS proxy")


@app.on_event("startup")
def _startup() -> None:
    global MODEL
    MODEL = load_model()


def _check_auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    expected = f"Bearer {API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL is not None, "device": str(DEVICE)}


@app.get("/v1/models")
def list_models() -> dict:
    return {
        "object": "list",
        "data": [
            {"id": "omnivoice", "object": "model", "owned_by": "k2-fsa"},
            {"id": "tts-1", "object": "model", "owned_by": "k2-fsa"},
            {"id": "tts-1-hd", "object": "model", "owned_by": "k2-fsa"},
        ],
    }


@app.get("/v1/audio/voices")
def list_voices() -> dict:
    # Hide "_"-prefixed template/comment entries.
    names = [k for k in VOICES.get("voices", {}) if not k.startswith("_")]
    return {"voices": names}


@app.post("/v1/audio/speech")
async def create_speech(
    req: SpeechRequest,
    authorization: Optional[str] = Header(default=None),
) -> Response:
    _check_auth(authorization)
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' must not be empty.")

    spec = resolve_voice(req.voice)
    # Explicit per-request overrides win over the voice preset.
    instruct = req.instruct if req.instruct is not None else spec.get("instruct")
    ref_audio = req.ref_audio if req.ref_audio is not None else spec.get("ref_audio")
    ref_text = req.ref_text if req.ref_text is not None else spec.get("ref_text")
    language = req.language if req.language is not None else spec.get("language")

    gen_kwargs = dict(GEN_DEFAULTS)
    gen_kwargs.update(
        text=req.input,
        language=language,
        instruct=instruct,
        ref_audio=ref_audio,
        ref_text=ref_text,
        speed=float(req.speed),
    )

    logger.info(
        "synth: %d chars | voice=%s instruct=%s ref=%s fmt=%s speed=%s",
        len(req.input),
        req.voice,
        instruct,
        bool(ref_audio),
        req.response_format,
        req.speed,
    )

    # Serialize GPU access; run blocking generate() off the event loop.
    async with _GEN_LOCK:
        try:
            audios = await asyncio.to_thread(MODEL.generate, **gen_kwargs)
        except ValueError as exc:
            # e.g. unsupported instruct items -> surface as a 400, not a 500.
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("generation failed")
            raise HTTPException(status_code=500, detail=f"generation failed: {exc}")

    wav = audios[0]
    data = encode_audio(wav, MODEL.sampling_rate, req.response_format)
    _, content_type = FORMAT_MAP[req.response_format.lower()]
    return Response(content=data, media_type=content_type)


@app.exception_handler(HTTPException)
async def _openai_error(_: Request, exc: HTTPException) -> JSONResponse:
    # Mirror OpenAI's error envelope so clients parse it cleanly.
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "invalid_request_error"}},
    )


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("OMNIVOICE_HOST", "0.0.0.0"),
        port=int(os.environ.get("OMNIVOICE_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
