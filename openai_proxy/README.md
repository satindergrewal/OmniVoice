# OmniVoice — OpenAI-compatible TTS proxy

A thin FastAPI server that exposes the OpenAI `POST /v1/audio/speech` API and
synthesizes with a local OmniVoice model. Any app that already talks to OpenAI
TTS works unchanged — just point its `base_url` here.

## Install

```bash
# from the repo root, with OmniVoice already installed (pip install -e .)
pip install -r openai_proxy/requirements.txt

# mp3/aac/opus output needs ffmpeg on PATH; wav/flac/pcm do not:
#   macOS:  brew install ffmpeg
```

## Run

```bash
python -m openai_proxy.app
# or
uvicorn openai_proxy.app:app --host 0.0.0.0 --port 8000
```

First request triggers model download/load (a few seconds to minutes the first time).

### Config (env vars)

| Var | Default | Purpose |
|-----|---------|---------|
| `OMNIVOICE_MODEL`  | `k2-fsa/OmniVoice` | HF repo id or local checkpoint path |
| `OMNIVOICE_DEVICE` | auto | `cuda` / `cpu` / `mps` / `xpu` |
| `OMNIVOICE_DTYPE`  | `float16` | `float16` / `bfloat16` / `float32` |
| `OMNIVOICE_VOICES` | `./voices.json` | voice → OmniVoice spec mapping |
| `OMNIVOICE_API_KEY`| _unset_ | if set, require `Authorization: Bearer <key>` |
| `OMNIVOICE_HOST` / `OMNIVOICE_PORT` | `0.0.0.0` / `8000` | bind address |

## Use it (OpenAI client, unchanged)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

client.audio.speech.create(
    model="omnivoice",
    voice="fable",
    input="[surprise-oh] You actually built this? [laughter] Incredible.",
    response_format="mp3",
).stream_to_file("out.mp3")
```

Or curl:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","voice":"nova","input":"Hello there [sigh] it works.","response_format":"wav"}' \
  --output out.wav
```

## What maps to what

| OpenAI field | OmniVoice behavior |
|--------------|--------------------|
| `input` | `text` — **non-verbal tags work here** (`[laughter]`, `[sigh]`, …) |
| `voice` | looked up in `voices.json` → an `instruct` preset or a voice-clone preset |
| `speed` | `speed` factor (passthrough) |
| `response_format` | `mp3` (default), `wav`, `flac`, `opus`, `aac`, `pcm` |
| `model` | ignored — there is only one local model |

`pcm` is raw signed-16-bit-LE mono at the model's sample rate (24 kHz).

## Voices

Edit `voices.json`. Each voice is one of two modes:

**Voice design** (no audio asset needed) — uses the `instruct` tag vocabulary
(gender / age / pitch / `whisper` / accent):

```json
"my_narrator": { "instruct": "male, british accent, low pitch" }
```

**Voice clone** (clone a real clip) — point at a reference wav + its transcript:

```json
"my_clone": {
  "ref_audio": "/abs/path/to/reference.wav",
  "ref_text": "Exact transcript of the reference clip.",
  "language": "English"
}
```

`GET /v1/audio/voices` lists the configured names. Unknown voices fall back to
`default_voice`.

### Per-request overrides (non-OpenAI extras)

Clients that support `extra_body` can override the preset per call:
`instruct`, `ref_audio`, `ref_text`, `language`.

```python
client.audio.speech.create(
    model="omnivoice", voice="alloy",
    input="Custom styled line.",
    extra_body={"instruct": "female, australian accent, high pitch"},
).stream_to_file("out.mp3")
```

## Notes / limits

- Requests are **serialized** (one GPU generation at a time) to keep VRAM
  bounded. For real concurrency, run multiple workers/processes behind a load
  balancer, each with its own model.
- Unsupported `instruct` items return a clean **400** with the valid-tag list —
  same validation as the core model.
- This proxy only bridges the API surface; it adds no new vocal expressions.
  Moans/breathiness still require a reference clip or a fine-tuned tag.
