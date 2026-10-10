"""Núcleo compartido entre la CLI y la app web.

Toda la lógica de carga de modelo, transcripción y generación de
archivos de salida vive acá, para no duplicarla entre interfaces.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from . import paths
from .cuda_env import patch_ld_library_path

# Debe correr ANTES de que CTranslate2 intente cargar CUDA (ver cuda_env.py
# para el porqué de esto). Como la carga es perezosa, alcanza con que esto
# se ejecute en algún punto antes de crear el WhisperModel.
PATCHED_CUDA_LIB_DIRS = patch_ld_library_path()

from faster_whisper import WhisperModel  # noqa: E402  (después del patch, a propósito)

OutputFormat = Literal["txt", "srt", "vtt", "json"]

DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_LANGUAGE = "es"
DEFAULT_DEVICE = "cuda"
DEFAULT_COMPUTE_TYPE = "float16"


def default_compute_type(device: str) -> str:
    """int8 anda en cualquier CPU; float16 necesita GPU con soporte para eso.

    Cualquier valor de `device` que no sea literalmente "cpu" (incluyendo
    "cuda" y "auto") asume que hay GPU de por medio y pide float16.

    Usado como fallback cuando no se especifica compute_type explícito,
    tanto al arrancar (CLI/web/daemon) como al cambiar de dispositivo en
    caliente desde la web — para no terminar pidiendo float16 en CPU y
    explotar con "Requested float16 compute type, but the target device
    ... do not support efficient float16 computation".
    """
    return "int8" if device == "cpu" else "float16"

# Dónde se descargan/cachean los pesos del modelo. Por defecto, dentro de
# $AI_MODELS/whisper (ver paths.py); se puede pisar por config o env var.
DEFAULT_DOWNLOAD_ROOT = str(paths.MODELS_DIR)

# Modelos curados para el selector de la web (podés usar cualquier otro
# nombre válido de faster-whisper vía --model / STT_MODEL, esto es solo
# para no dejar el selector vacío).
AVAILABLE_MODELS = [
    {"id": "distil-large-v3", "label": "distil-large-v3 (rápido, buen balance)"},
    {"id": "large-v3", "label": "large-v3 (máxima calidad)"},
    {"id": "large-v3-turbo", "label": "large-v3-turbo (rápido, calidad alta)"},
    {"id": "medium", "label": "medium"},
    {"id": "small", "label": "small (liviano)"},
    {"id": "base", "label": "base (muy liviano)"},
    {"id": "tiny", "label": "tiny (mínimo, para pruebas)"},
]

# Idiomas más comunes. faster-whisper soporta la lista completa de Whisper
# (~99 idiomas); si necesitás uno que no está acá, se puede pasar el
# código ISO-639-1 igual vía --language / STT_LANGUAGE.
AVAILABLE_LANGUAGES = [
    {"id": "", "label": "Autodetectar"},
    {"id": "es", "label": "Español"},
    {"id": "en", "label": "Inglés"},
    {"id": "pt", "label": "Portugués"},
    {"id": "fr", "label": "Francés"},
    {"id": "de", "label": "Alemán"},
    {"id": "it", "label": "Italiano"},
    {"id": "ja", "label": "Japonés"},
    {"id": "ko", "label": "Coreano"},
    {"id": "zh", "label": "Chino"},
    {"id": "ru", "label": "Ruso"},
    {"id": "ar", "label": "Árabe"},
    {"id": "nl", "label": "Neerlandés"},
    {"id": "tr", "label": "Turco"},
    {"id": "pl", "label": "Polaco"},
    {"id": "ca", "label": "Catalán"},
]


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    text: str
    segments: list[Segment]
    language: str
    duration: float
    elapsed: float
    source_path: Path

    @property
    def speed_factor(self) -> float | None:
        if self.elapsed <= 0:
            return None
        return self.duration / self.elapsed


@dataclass
class TranscriberConfig:
    model_name: str = DEFAULT_MODEL
    device: str = DEFAULT_DEVICE
    compute_type: str = DEFAULT_COMPUTE_TYPE
    language: str | None = DEFAULT_LANGUAGE
    beam_size: int = 5
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    download_root: str | None = DEFAULT_DOWNLOAD_ROOT
    task: Literal["transcribe", "translate"] = "transcribe"

_LANGUAGE_UNSET = object()
_TASK_UNSET = object()

class Transcriber:
    """Envoltorio delgado sobre WhisperModel, cargado una sola vez."""

    def __init__(self, config: TranscriberConfig | None = None):
        self.config = config or TranscriberConfig()
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = self._load()
        return self._model

    def _load(self) -> WhisperModel:
        cfg = self.config
        return WhisperModel(
            cfg.model_name,
            device=cfg.device,
            compute_type=cfg.compute_type,
            download_root=cfg.download_root,
        )

    def transcribe(self, path: str | Path, language=_LANGUAGE_UNSET, task=_TASK_UNSET) -> TranscriptionResult:
        """Transcribe un archivo.

        `language` permite pisar el idioma configurado para esta llamada
        puntual (usado por la web, para que el selector de idioma no
        requiera recargar el modelo). Si no se pasa, se usa
        `self.config.language`. Pasar `None` explícitamente pide
        autodetección de idioma.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe: {path}")

        cfg = self.config
        effective_language = cfg.language if language is _LANGUAGE_UNSET else language
        effective_task = cfg.task if task is _TASK_UNSET else task
        start_time = time.time()

        raw_segments, info = self.model.transcribe(
            str(path),
            language=effective_language,
            task=effective_task,
            beam_size=cfg.beam_size,
            vad_filter=cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": cfg.min_silence_duration_ms},
        )

        segments = [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in raw_segments
        ]

        elapsed = time.time() - start_time
        full_text = "\n".join(s.text for s in segments)

        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=info.language,
            duration=info.duration,
            elapsed=elapsed,
            source_path=path,
        )


# ------------------------------------------------------------------
# Formato de tiempo
# ------------------------------------------------------------------

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def format_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_vtt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# ------------------------------------------------------------------
# Serialización a distintos formatos
# ------------------------------------------------------------------

def to_srt(segments: Iterable[Segment]) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: Iterable[Segment]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{format_vtt_time(seg.start)} --> {format_vtt_time(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def to_timestamped_txt(segments: Iterable[Segment]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{format_time(seg.start)} --> {format_time(seg.end)}] {seg.text}")
    return "\n".join(lines)


def save_result(
    result: TranscriptionResult,
    output_dir: str | Path,
    formats: Iterable[OutputFormat] = ("txt", "srt", "json"),
) -> dict[str, Path]:
    """Guarda el resultado en los formatos pedidos. Devuelve un dict formato -> ruta."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result.source_path.stem

    saved: dict[str, Path] = {}

    if "txt" in formats:
        p = output_dir / f"{stem}.txt"
        p.write_text(result.text, encoding="utf-8")
        saved["txt"] = p

    if "srt" in formats:
        p = output_dir / f"{stem}.srt"
        p.write_text(to_srt(result.segments), encoding="utf-8")
        saved["srt"] = p

    if "vtt" in formats:
        p = output_dir / f"{stem}.vtt"
        p.write_text(to_vtt(result.segments), encoding="utf-8")
        saved["vtt"] = p

    if "json" in formats:
        import json

        p = output_dir / f"{stem}.json"
        payload = {
            "text": result.text,
            "language": result.language,
            "duration": result.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in result.segments
            ],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        saved["json"] = p

    # el timestamped txt siempre se guarda como extra, es útil para revisar
    p = output_dir / f"{stem}_timestamps.txt"
    p.write_text(to_timestamped_txt(result.segments), encoding="utf-8")
    saved["timestamps"] = p

    return saved
