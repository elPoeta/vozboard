"""Listener de atajo global multiplataforma-Linux vía evdev.

No usa APIs de X11 ni de ningún compositor Wayland en particular: lee
directamente los eventos crudos del kernel (/dev/input/eventX), así que
funciona igual en X11, en cualquier compositor Wayland (KWin, Mutter,
wlroots...) y hasta en una consola virtual. Es el mismo principio que usa
ydotool para escribir, pero al revés — acá solo escuchamos, no inyectamos
nada.

No se hace un grab exclusivo del dispositivo (no se llama a EVIOCGRAB),
así que las teclas siguen llegando normalmente a la aplicación con foco;
solo estamos "espiando" para detectar la combinación.

Requiere poder leer /dev/input/event*. En una sesión de escritorio local
esto normalmente ya funciona solo (systemd-logind le da ACL de los
dispositivos de la sesión activa al usuario logueado). Si no:

    sudo usermod -aG input $USER   # y volver a iniciar sesión
"""

from __future__ import annotations

import threading
import time
from typing import Callable

try:
    import evdev
    from evdev import ecodes
except ImportError:  # evdev es una dependencia opcional (solo para stt-daemon)
    evdev = None
    ecodes = None


# Combo por default: Ctrl derecho + Alt derecho, sostenidos juntos.
# Elegido por ser poco común como atajo de otras apps. Cambiar con
# STT_HOTKEY="KEY_LEFTCTRL+KEY_LEFTALT+KEY_SPACE" (por ejemplo), separado
# por '+'. Nombres válidos: cualquier KEY_* de evdev.ecodes. Usá
# `uv run stt-daemon --list-keys` para descubrir los nombres de tu teclado.
DEFAULT_COMBO = ("KEY_RIGHTCTRL", "KEY_RIGHTALT")

# Segundo atajo, para dictar directo con pegado tipo "terminal"
# (Ctrl+Shift+V en vez de Ctrl+V — ver inject.py). Lado izquierdo del
# teclado a propósito, para no pisarse con el de arriba. Cambiar con
# STT_HOTKEY_TERMINAL.
DEFAULT_TERMINAL_COMBO = ("KEY_LEFTCTRL", "KEY_LEFTALT")


class HotkeyListener:
    def __init__(
        self,
        combo: tuple[str, ...] = DEFAULT_COMBO,
        on_trigger: Callable[[], None] | None = None,
    ):
        self.combo = set(combo)
        self.on_trigger = on_trigger

        self._held: set[str] = set()
        self._fired = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []

    def _keyboards(self) -> list["evdev.InputDevice"]:
        devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
        keyboards = []
        for d in devices:
            caps = d.capabilities()
            keys = caps.get(ecodes.EV_KEY, [])
            # Heurística estándar para distinguir teclados de otros
            # dispositivos con EV_KEY (mouses con botones, mandos, etc.):
            # un teclado real tiene la tecla 'A'.
            if ecodes.EV_KEY in caps and ecodes.KEY_A in keys:
                keyboards.append(d)
        return keyboards

    def _watch(self, device: "evdev.InputDevice") -> None:
        try:
            for event in device.read_loop():
                if self._stop_event.is_set():
                    return
                if event.type != ecodes.EV_KEY:
                    continue

                key_event = evdev.categorize(event)
                key_name = key_event.keycode
                if isinstance(key_name, list):
                    key_name = key_name[0]

                with self._lock:
                    if key_event.keystate == key_event.key_down:
                        self._held.add(key_name)
                    elif key_event.keystate == key_event.key_up:
                        self._held.discard(key_name)
                        self._fired = False

                    if self.combo.issubset(self._held) and not self._fired:
                        self._fired = True
                        if self.on_trigger:
                            self.on_trigger()
        except OSError:
            return  # el dispositivo desapareció (desconexión, suspensión, etc.)

    def start(self) -> None:
        if evdev is None:
            raise RuntimeError(
                "Falta el paquete 'evdev'. Instalalo con: uv sync --extra daemon"
            )
        keyboards = self._keyboards()
        if not keyboards:
            raise RuntimeError(
                "No se encontró ningún teclado en /dev/input con permisos de "
                "lectura. Probá: sudo usermod -aG input $USER (y reiniciar sesión)."
            )
        for device in keyboards:
            t = threading.Thread(target=self._watch, args=(device,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop_event.set()


def debug_print_keys(seconds: float = 20) -> None:
    """Imprime los nombres de tecla en tiempo real, para armar STT_HOTKEY.

    Usado por `stt-daemon --list-keys`.
    """
    if evdev is None:
        raise RuntimeError(
            "Falta el paquete 'evdev'. Instalalo con: uv sync --extra daemon"
        )

    devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
    keyboards = [
        d
        for d in devices
        if ecodes.EV_KEY in d.capabilities()
        and ecodes.KEY_A in d.capabilities().get(ecodes.EV_KEY, [])
    ]
    if not keyboards:
        print("No se encontró ningún teclado accesible en /dev/input.")
        return

    names = ", ".join(d.name for d in keyboards)
    print(f"Escuchando {len(keyboards)} dispositivo(s) [{names}] durante {seconds:.0f}s.")
    print("Presioná las teclas que querés usar como atajo (Ctrl+C para cortar antes)...")
    print()

    stop_event = threading.Event()

    def _watch(device: "evdev.InputDevice") -> None:
        try:
            for event in device.read_loop():
                if stop_event.is_set():
                    return
                if event.type != ecodes.EV_KEY:
                    continue
                key_event = evdev.categorize(event)
                if key_event.keystate == key_event.key_down:
                    name = key_event.keycode
                    if isinstance(name, list):
                        name = name[0]
                    print(f"  {name}")
        except OSError:
            return

    threads = [threading.Thread(target=_watch, args=(d,), daemon=True) for d in keyboards]
    for t in threads:
        t.start()

    try:
        time.sleep(seconds)
    except KeyboardInterrupt:
        pass
    stop_event.set()
