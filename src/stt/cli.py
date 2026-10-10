"""CLI de vozboard.

Uso típico con uv:

    uv run stt file audio.mp3
    uv run stt record
    uv run stt menu
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import paths
from .core import (
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    PATCHED_CUDA_LIB_DIRS,
    Transcriber,
    TranscriberConfig,
    default_compute_type,
    save_result,
)
from .recorder import record_from_microphone

AUDIO_DIR = paths.AUDIO_DIR
OUTPUT_DIR = paths.OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stt", description="Vozboard local con GPU")

    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Modelo Whisper (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["cuda", "cpu", "auto"])
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Precisión (float16, int8, etc.). Default automático según --device.",
    )
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Código de idioma, ej. es, en. Vacío para autodetectar.")
    parser.add_argument(
        "--formats",
        default="txt,srt,json",
        help="Formatos de salida separados por coma: txt,srt,vtt,json",
    )
    parser.add_argument(
        "--download-root",
        default=None,
        help=f"Carpeta donde se descargan los pesos del modelo (default: {paths.MODELS_DIR})",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("record", help="Grabar desde el micrófono y transcribir")

    file_parser = subparsers.add_parser("file", help="Transcribir un archivo de audio")
    file_parser.add_argument("path", help="Ruta al archivo de audio (wav, mp3, m4a, flac, ogg...)")

    subparsers.add_parser("menu", help="Menú interactivo (estilo clásico)")

    return parser


def make_config(args: argparse.Namespace) -> TranscriberConfig:
    compute_type = args.compute_type or default_compute_type(args.device)
    config = TranscriberConfig(
        model_name=args.model,
        device=args.device,
        compute_type=compute_type,
        language=args.language or None,
    )
    if args.download_root:
        config.download_root = args.download_root
    return config


def _print_result(result) -> None:
    print()
    print("-" * 60)
    print("RESULTADO")
    print("-" * 60)
    print()
    print(result.text)
    print()
    print("-" * 60)
    print(f"Duración audio : {result.duration:.2f} segundos")
    print(f"Tiempo proceso : {result.elapsed:.2f} segundos")
    if result.speed_factor:
        print(f"Velocidad      : {result.speed_factor:.2f}x tiempo real")
    print()


def run_transcription(transcriber: Transcriber, path: Path, formats: list[str]) -> None:
    print()
    print(f"Archivo: {path}")
    print("Procesando con GPU..." if transcriber.config.device == "cuda" else "Procesando...")

    result = transcriber.transcribe(path)
    _print_result(result)

    saved = save_result(result, OUTPUT_DIR, formats=formats)
    print("Archivos guardados:")
    for fmt, p in saved.items():
        print(f"  {fmt:11s} -> {p}")
    print()


def cmd_record(transcriber: Transcriber, formats: list[str]) -> None:
    filename = record_from_microphone(AUDIO_DIR)
    if filename:
        run_transcription(transcriber, filename, formats)


def cmd_file(transcriber: Transcriber, path: str, formats: list[str]) -> None:
    run_transcription(transcriber, Path(path.strip().strip('"').strip("'")), formats)


def cmd_menu(transcriber: Transcriber, formats: list[str]) -> None:
    while True:
        print()
        print("=" * 60)
        print("             VOZBOARD")
        print("=" * 60)
        print()
        print("  1. 🎙️  Grabar desde micrófono")
        print("  2. 📁 Transcribir archivo")
        print("  3. ❌ Salir")
        print()
        option = input("Seleccione una opción: ").strip()

        if option == "1":
            cmd_record(transcriber, formats)
        elif option == "2":
            print()
            print("Introduce la ruta del archivo de audio:")
            path = input("> ").strip()
            cmd_file(transcriber, path, formats)
        elif option == "3":
            print()
            print("Hasta luego.")
            print()
            break
        else:
            print()
            print("Opción inválida.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    paths.ensure_dirs()

    config = make_config(args)

    print()
    print("Cargando Whisper...")
    print(f"Modelo:      {config.model_name}")
    print(f"Dispositivo: {config.device}")
    print(f"Precisión:   {config.compute_type}")
    print(f"Pesos en:    {config.download_root}")
    print(f"Audio en:    {AUDIO_DIR}")
    print(f"Salida en:   {OUTPUT_DIR}")
    if not paths.AI_HOME:
        print("(AI_HOME no está definido en el entorno: usando carpetas locales del proyecto)")
    if PATCHED_CUDA_LIB_DIRS:
        print("LD_LIBRARY_PATH ajustado con librerías CUDA de pip:")
        for d in PATCHED_CUDA_LIB_DIRS:
            print(f"  - {d}")
    print()

    transcriber = Transcriber(config)

    try:
        if args.command == "record":
            cmd_record(transcriber, formats)
        elif args.command == "file":
            cmd_file(transcriber, args.path, formats)
        elif args.command == "menu":
            cmd_menu(transcriber, formats)
    except KeyboardInterrupt:
        print()
        print("Aplicación finalizada.")
    except FileNotFoundError as e:
        print()
        print(f"ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
