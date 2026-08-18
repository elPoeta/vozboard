"""Daemon standalone: ícono de bandeja + atajo global + dictado directo.

Flujo: presionás un atajo -> empieza a grabar -> lo presionás de nuevo
-> para, transcribe, y escribe el resultado en la ventana (o terminal)
con foco — como VoiceInk/MacWhisper, pero para Linux.

Hay dos atajos por default, para las dos formas de pegado (ver
inject.py sobre por qué):

    Ctrl derecho + Alt derecho  -> pega con Ctrl+V        (apps normales)
    Ctrl izquierdo + Alt izquierdo -> pega con Ctrl+Shift+V  (terminales)

El ícono de bandeja también tiene un menú para cambiar cuál de los dos
modos es el "default" (usado al hacer clic en el ícono en vez de usar
un atajo).

Uso:

    uv run stt-daemon                          # arranca el daemon
    uv run stt-daemon --list-keys              # ayuda para elegir atajos
    uv run stt-daemon --device cpu --model medium
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import paths
from .core import (
    DEFAULT_COMPUTE_TYPE,
    DEFAULT_DEVICE,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    Transcriber,
    TranscriberConfig,
    default_compute_type,
)
from .hotkey import DEFAULT_COMBO, DEFAULT_TERMINAL_COMBO, HotkeyListener
from .inject import TextInjector, ydotool_available
from .recorder import Recorder
from .tray import Status, TrayApp


def _config_from_env() -> TranscriberConfig:
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


def _combo_from_env(var: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(var)
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split("+") if part.strip())


class Orchestrator:
    """Conecta grabación, transcripción e inyección de texto. Se llama
    desde los hilos de los atajos globales Y desde el hilo principal de
    Qt (clic en el ícono), por eso el lock."""

    def __init__(self, tray: TrayApp, transcriber: Transcriber, injector: TextInjector):
        self.tray = tray
        self.transcriber = transcriber
        self.injector = injector
        self.recorder = Recorder()
        self.recording = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._pending_terminal = False
        self.default_terminal_paste = False

    def set_default_paste_mode(self, terminal: bool) -> None:
        self.default_terminal_paste = terminal
        self.tray.set_default_paste_mode(terminal)

    def toggle(self, terminal: bool | None = None) -> None:
        """`terminal=None` (clic en el ícono, o acción de menú) usa el
        modo por default configurado. `terminal=True/False` explícito
        (los dos atajos de teclado) lo pisa para esta grabación puntual."""
        with self._lock:
            if not self.recording:
                self._start(terminal)
            else:
                self._stop_and_process()

    def _start(self, terminal: bool | None) -> None:
        if terminal is None:
            terminal = self.default_terminal_paste
        self.recorder.start()
        self.recording = True
        self._pending_terminal = terminal
        self.tray.set_status(Status.RECORDING)

    def _stop_and_process(self) -> None:
        self.recording = False
        filename = self.recorder.stop(paths.AUDIO_DIR)

        if filename is None:
            self.tray.set_status(Status.IDLE)
            return

        self.tray.set_status(Status.PROCESSING)
        self.executor.submit(self._process, filename, self._pending_terminal)

    def _process(self, filename: Path, terminal: bool) -> None:
        try:
            result = self.transcriber.transcribe(filename)
            text = result.text.strip()
            if text:
                self.injector.inject(text, terminal=terminal)
                mode_label = "Ctrl+Shift+V (terminal)" if terminal else "Ctrl+V"
                self.tray.notify("Transcripción pegada", f"Modo: {mode_label}")
            self.tray.set_status(Status.IDLE)
        except Exception as e:
            self.tray.notify("Error de transcripción", str(e), is_error=True)
            self.tray.set_status(Status.ERROR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stt-daemon")
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Escucha el teclado e imprime nombres de tecla, para armar STT_HOTKEY. No inicia el daemon.",
    )
    parser.add_argument("--model", default=None, help="O STT_MODEL")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"], help="O STT_DEVICE")
    parser.add_argument("--compute-type", default=None, help="O STT_COMPUTE_TYPE")
    parser.add_argument(
        "--language", default=None, help="O STT_LANGUAGE (vacío para autodetectar)"
    )
    parser.add_argument("--download-root", default=None, help="O STT_DOWNLOAD_ROOT")
    parser.add_argument(
        "--hotkey",
        default=None,
        help="Atajo para pegado normal, ej. KEY_RIGHTCTRL+KEY_RIGHTALT (o STT_HOTKEY)",
    )
    parser.add_argument(
        "--hotkey-terminal",
        default=None,
        help="Atajo para pegado tipo terminal, ej. KEY_LEFTCTRL+KEY_LEFTALT (o STT_HOTKEY_TERMINAL)",
    )
    parser.add_argument(
        "--inject-mode",
        default=None,
        choices=["paste", "type"],
        help="O STT_INJECT_MODE (default: paste)",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.list_keys:
        from .hotkey import debug_print_keys

        debug_print_keys()
        return

    # Los flags pisan las env vars correspondientes; _config_from_env() y
    # las funciones de combo las leen más abajo.
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
    if args.hotkey:
        os.environ["STT_HOTKEY"] = args.hotkey
    if args.hotkey_terminal:
        os.environ["STT_HOTKEY_TERMINAL"] = args.hotkey_terminal
    if args.inject_mode:
        os.environ["STT_INJECT_MODE"] = args.inject_mode

    if not ydotool_available():
        print(
            "AVISO: no se encontró 'ydotool' en el PATH. El dictado no va a "
            "poder escribir en la ventana activa hasta que lo instales "
            "(ver README, sección 'Modo standalone')."
        )

    # PySide6 se importa recién acá para que --list-keys no dependa de Qt.
    from PySide6.QtWidgets import QApplication

    paths.ensure_dirs()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    tray = TrayApp(app)

    print("Cargando modelo...")
    transcriber = Transcriber(_config_from_env())
    _ = transcriber.model  # forzar la carga ahora, no en el primer atajo
    print("Modelo cargado. Listo.")

    injector = TextInjector(mode=os.environ.get("STT_INJECT_MODE", "paste"))
    orchestrator = Orchestrator(tray, transcriber, injector)

    tray.request_toggle.connect(orchestrator.toggle)
    tray.request_quit.connect(app.quit)
    tray.request_set_paste_mode.connect(orchestrator.set_default_paste_mode)

    combo = _combo_from_env("STT_HOTKEY", DEFAULT_COMBO)
    combo_terminal = _combo_from_env("STT_HOTKEY_TERMINAL", DEFAULT_TERMINAL_COMBO)
    combo_label = " + ".join(combo)
    combo_terminal_label = " + ".join(combo_terminal)

    print(f"Atajo (Ctrl+V):          {combo_label}")
    print(f"Atajo (Ctrl+Shift+V):    {combo_terminal_label}")

    listener = HotkeyListener(combo=combo, on_trigger=lambda: orchestrator.toggle(terminal=False))
    listener_terminal = HotkeyListener(
        combo=combo_terminal, on_trigger=lambda: orchestrator.toggle(terminal=True)
    )

    started = []
    for label, hk in (("normal", listener), ("terminal", listener_terminal)):
        try:
            hk.start()
            started.append(label)
        except RuntimeError as e:
            print(f"AVISO (atajo {label}): {e}")
            tray.notify(f"Atajo '{label}' no disponible", str(e), is_error=True)

    if started:
        tray.notify(
            "Vozboard",
            f"Listo.\n{combo_label} → Ctrl+V\n{combo_terminal_label} → Ctrl+Shift+V (terminal)",
        )

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
