"""App web de Vozboard.

Uso con uv:

    uv run stt-web
    # o directamente:
    uv run uvicorn stt.web:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import paths
from .core import (
    AVAILABLE_LANGUAGES,
    AVAILABLE_MODELS,
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    PATCHED_CUDA_LIB_DIRS,
    Transcriber,
    TranscriberConfig,
    default_compute_type,
    save_result,
)
from .core import _LANGUAGE_UNSET

BASE_DIR = Path(__file__).resolve().parent.parent.parent
AUDIO_DIR = paths.AUDIO_DIR
OUTPUT_DIR = paths.OUTPUT_DIR
WEB_DIR = BASE_DIR / "web"

paths.ensure_dirs()

_transcriber: Transcriber | None = None
_reload_lock = asyncio.Lock()  # evita transcribir mientras se recarga el modelo


def get_config() -> TranscriberConfig:
    device = os.environ.get("STT_DEVICE", DEFAULT_DEVICE)
    compute_type = os.environ.get("STT_COMPUTE_TYPE") or default_compute_type(device)
    config = TranscriberConfig(
        model_name=os.environ.get("STT_MODEL", DEFAULT_MODEL),
        device=device,
        compute_type=compute_type,
        language=os.environ.get("STT_LANGUAGE", DEFAULT_LANGUAGE) or None,
    )
    if os.environ.get("STT_DOWNLOAD_ROOT"):
        config.download_root = os.environ["STT_DOWNLOAD_ROOT"]
    return config


def _print_banner(config: TranscriberConfig, patched_dirs: list[str]) -> None:
    print()
    print(f"Modelo:      {config.model_name}")
    print(f"Dispositivo: {config.device} ({config.compute_type})")
    print(f"Pesos en:    {config.download_root}")
    print(f"Audio en:    {AUDIO_DIR}")
    print(f"Salida en:   {OUTPUT_DIR}")
    if not paths.AI_HOME:
        print("(AI_HOME no está definido: usando carpetas locales del proyecto)")
    if patched_dirs:
        print("LD_LIBRARY_PATH ajustado con librerías CUDA de pip:")
        for d in patched_dirs:
            print(f"  - {d}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _transcriber
    config = get_config()
    _print_banner(config, PATCHED_CUDA_LIB_DIRS)

    print("Cargando modelo...")
    _transcriber = Transcriber(config)
    # Cargar el modelo al arrancar, no en el primer request.
    _ = _transcriber.model
    print("Modelo cargado. Listo.")
    print()
    yield


app = FastAPI(title="Vozboard", lifespan=lifespan)


@app.get("/api/status")
def status():
    cfg = _transcriber.config if _transcriber else get_config()
    return {
        "model": cfg.model_name,
        "device": cfg.device,
        "compute_type": cfg.compute_type,
        "language": cfg.language,
        "ready": _transcriber is not None,
        "paths": paths.describe(),
    }


@app.get("/api/models")
def list_models():
    return {"models": AVAILABLE_MODELS}


@app.get("/api/languages")
def list_languages():
    return {"languages": AVAILABLE_LANGUAGES}


class ConfigUpdate(BaseModel):
    model: str | None = None
    device: str | None = None
    compute_type: str | None = None


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    """Recarga el modelo con un nuevo model/device/compute_type.

    Esto es lo único que necesita recargar el modelo en memoria; el
    idioma se maneja aparte, por request, sin tocar esto (ver
    /api/transcribe).
    """
    global _transcriber

    if _transcriber is None:
        raise HTTPException(status_code=503, detail="El modelo todavía está cargando.")

    current = _transcriber.config
    new_model = update.model or current.model_name
    new_device = update.device or current.device
    new_compute_type = update.compute_type or (
        default_compute_type(new_device) if update.device else current.compute_type
    )

    if (new_model, new_device, new_compute_type) == (
        current.model_name,
        current.device,
        current.compute_type,
    ):
        return {"reloaded": False, **status()}

    new_config = TranscriberConfig(
        model_name=new_model,
        device=new_device,
        compute_type=new_compute_type,
        language=current.language,
        download_root=current.download_root,
    )

    async with _reload_lock:
        print(f"Recargando modelo: {new_model} / {new_device} / {new_compute_type}...")
        new_transcriber = Transcriber(new_config)
        try:
            # Cargar en un thread para no bloquear el loop de asyncio.
            await asyncio.to_thread(lambda: new_transcriber.model)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"No se pudo cargar el modelo: {e}")
        _transcriber = new_transcriber
        print("Modelo recargado.")

    return {"reloaded": True, **status()}


@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str | None = Form(None)):
    if _transcriber is None:
        raise HTTPException(status_code=503, detail="El modelo todavía está cargando.")

    suffix = Path(file.filename or "audio").suffix or ".wav"
    stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    dest = AUDIO_DIR / f"{stem}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # "" desde el selector significa autodetectar (None para faster-whisper).
    lang_override = (language or None) if language is not None else _LANGUAGE_UNSET

    async with _reload_lock:  # espera si hay una recarga de modelo en curso
        try:
            result = await asyncio.to_thread(_transcriber.transcribe, dest, lang_override)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    saved = save_result(result, OUTPUT_DIR, formats=("txt", "srt", "vtt", "json"))

    return JSONResponse(
        {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "elapsed": result.elapsed,
            "speed_factor": result.speed_factor,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text} for s in result.segments
            ],
            "downloads": {fmt: f"/api/output/{p.name}" for fmt, p in saved.items()},
        }
    )


@app.get("/api/output/{filename}")
def download(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or path.parent != OUTPUT_DIR:
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return FileResponse(path)


# Sirve la interfaz estática (index.html, etc.) en la raíz.
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="stt-web")
    parser.add_argument("--model", default=None, help="Modelo Whisper (o STT_MODEL)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="O STT_DEVICE")
    parser.add_argument("--compute-type", default=None, help="O STT_COMPUTE_TYPE")
    parser.add_argument("--language", default=None, help="Código de idioma, o vacío para autodetectar (o STT_LANGUAGE)")
    parser.add_argument("--download-root", default=None, help="O STT_DOWNLOAD_ROOT")
    parser.add_argument("--host", default=None, help="O STT_HOST (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="O STT_PORT (default 8000)")
    args = parser.parse_args()

    # Los flags pisan las env vars correspondientes; get_config() las lee
    # más abajo (en el lifespan de FastAPI), así que alcanza con setearlas acá.
    if args.model:
        os.environ["STT_MODEL"] = args.model
    if args.device:
        os.environ["STT_DEVICE"] = args.device
    if args.compute_type:
        os.environ["STT_COMPUTE_TYPE"] = args.compute_type
    if args.language is not None:
        os.environ["STT_LANGUAGE"] = args.language
    if args.download_root:
        os.environ["STT_DOWNLOAD_ROOT"] = args.download_root
    if args.host:
        os.environ["STT_HOST"] = args.host
    if args.port:
        os.environ["STT_PORT"] = str(args.port)

    uvicorn.run(
        "stt.web:app",
        host=os.environ.get("STT_HOST", "0.0.0.0"),
        port=int(os.environ.get("STT_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
