"""Resolución de rutas a partir del entorno de IA del usuario.

Si en el shell están definidas las variables de tu perfil:

    AI_HOME, AI_MODELS, AI_OUTPUT, AI_CACHE, HF_HOME, TORCH_HOME

las usamos para que los modelos de Whisper, el audio grabado/subido y las
transcripciones queden ordenados junto al resto de tu stack de IA, en vez
de esparcirse por `~/.cache` o carpetas locales sueltas.

Si esas variables no están definidas (por ejemplo corriendo en otra
máquina sin ese perfil), todo cae de vuelta a carpetas locales dentro del
proyecto, así que el proyecto sigue funcionando igual.
"""

from __future__ import annotations

import os
from pathlib import Path

AI_HOME = os.environ.get("AI_HOME")
AI_MODELS = os.environ.get("AI_MODELS") or (f"{AI_HOME}/models" if AI_HOME else None)
AI_OUTPUT = os.environ.get("AI_OUTPUT") or (f"{AI_HOME}/output" if AI_HOME else None)
AI_CACHE = os.environ.get("AI_CACHE") or (f"{AI_HOME}/cache" if AI_HOME else None)

# HF_HOME ya lo respeta huggingface_hub (y por lo tanto faster-whisper) por
# su cuenta si está exportado en el shell. Acá solo lo leemos para poder
# mostrarlo en el status / banner de arranque.
HF_HOME = os.environ.get("HF_HOME")

APP_NAME = "vozboard"


def _resolve(base: str | None, *parts: str, fallback: str) -> Path:
    if base:
        return Path(base, *parts).expanduser()
    return Path(fallback)


# Los modelos de Whisper (varios GB) van en su propia subcarpeta dentro de
# AI_MODELS, separados de otros modelos (llama.cpp, stable-diffusion, etc).
MODELS_DIR = _resolve(AI_MODELS, "whisper", fallback=".cache/whisper")

# Audio grabado por micrófono o subido desde la web.
AUDIO_DIR = _resolve(AI_OUTPUT, APP_NAME, "audio", fallback="audio")

# Transcripciones generadas (.txt/.srt/.vtt/.json).
OUTPUT_DIR = _resolve(AI_OUTPUT, APP_NAME, "transcripts", fallback="output")

# Cache de conversión de modelos CTranslate2, si hiciera falta.
CACHE_DIR = _resolve(AI_CACHE, APP_NAME, fallback=".cache/vozboard")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, AUDIO_DIR, OUTPUT_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, str]:
    """Para mostrar en el banner de arranque / endpoint de status."""
    return {
        "models_dir": str(MODELS_DIR),
        "audio_dir": str(AUDIO_DIR),
        "output_dir": str(OUTPUT_DIR),
        "cache_dir": str(CACHE_DIR),
        "hf_home": HF_HOME or "(no configurado, usa ~/.cache/huggingface)",
        "using_ai_profile": bool(AI_HOME),
    }
