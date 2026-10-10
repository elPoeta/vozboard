"""Inyección de texto en la ventana con foco, vía ydotool.

ydotool crea un teclado/mouse virtual a nivel de kernel (uinput), así que
funciona igual en X11 y en cualquier compositor Wayland — no depende de
protocolos específicos de KWin/Mutter/wlroots. Es la misma pieza que usan
proyectos como waystt para este mismo caso de uso (dictado -> texto en
la ventana activa).

Requiere:
  - el paquete `ydotool` instalado (Fedora: sudo dnf install ydotool)
  - el daemon `ydotoold` corriendo (ver README, sección "Modo standalone")
  - el usuario con acceso a /dev/uinput (grupo `input`, o una regla udev)

## Por qué el modo por default es "paste" y no "type"

`ydotool type` simula cada tecla una por una, y tiene un bug conocido
(abierto desde hace años, no arreglado a la fecha) donde el texto se
corta apenas aparece un carácter acentuado — é, á, í, ó, ú, ñ. Para
transcripción en español esto no es cosmético, es directamente que se
pierde la mitad del texto.

El modo "paste" lo evita: copia el texto (UTF-8 completo, sin pasar por
simulación de teclas) al portapapeles, y solo simula un atajo de teclado
— códigos de tecla ASCII, ajenos al bug. Es también más rápido para
textos largos.

## Ctrl+V vs Ctrl+Shift+V (apps normales vs terminales)

La mayoría de las apps GTK/Qt pegan con Ctrl+V. La mayoría de las
terminales (Konsole, GNOME Terminal, etc.) NO usan Ctrl+V para pegar —
está reservado o simplemente no hace nada — y en cambio usan
Ctrl+Shift+V. No hay forma universal de saber desde acá qué tipo de
ventana tiene el foco sin meterse con APIs específicas de cada
compositor (lo cual rompería el objetivo de ser agnóstico), así que
`TextInjector.inject()` recibe un flag `terminal: bool` explícito — lo
decide quién llama (por ejemplo, dos atajos de teclado distintos: ver
daemon.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess

# Códigos de tecla Linux (ver /usr/include/linux/input-event-codes.h)
_KEY_LEFTCTRL = 29
_KEY_LEFTSHIFT = 42
_KEY_V = 47

# Formato ydotool: "<código>:<1=down,0=up>"
PASTE_KEYS_NORMAL = [f"{_KEY_LEFTCTRL}:1", f"{_KEY_V}:1", f"{_KEY_V}:0", f"{_KEY_LEFTCTRL}:0"]
PASTE_KEYS_TERMINAL = [
    f"{_KEY_LEFTCTRL}:1",
    f"{_KEY_LEFTSHIFT}:1",
    f"{_KEY_V}:1",
    f"{_KEY_V}:0",
    f"{_KEY_LEFTSHIFT}:0",
    f"{_KEY_LEFTCTRL}:0",
]


class InjectionError(RuntimeError):
    pass


def ydotool_available() -> bool:
    return shutil.which("ydotool") is not None


def _run(
    cmd: list[str],
    input_text: str | None = None,
    capture: bool = True,
    timeout: float = 30,
) -> None:
    """Ejecuta un comando externo.

    `capture=False` es crítico para herramientas que se demonizan a sí
    mismas (como wl-copy): si les capturamos stdout/stderr con PIPE, el
    proceso hijo que se queda corriendo en segundo plano hereda esos
    pipes abiertos, y `subprocess.run` se cuelga esperando a que se
    cierren — cosa que nunca pasa hasta que matás el proceso. Con
    stdout/stderr apuntando a /dev/null en cambio, `subprocess.run`
    vuelve apenas el proceso *padre* (el que hace el fork) termina, que
    es casi instantáneo.
    """
    try:
        subprocess.run(
            cmd,
            input=input_text,
            check=True,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise InjectionError(
            f"No se encontró '{cmd[0]}'. ¿Está instalado? (dnf install {cmd[0]})"
        )
    except subprocess.CalledProcessError as e:
        detail = ((e.stderr or "") + (e.stdout or "")).strip() if capture else ""
        suffix = f": {detail}" if detail else " (sin salida, ver logs del sistema)"
        raise InjectionError(f"'{' '.join(cmd)}' falló{suffix}")
    except subprocess.TimeoutExpired:
        raise InjectionError(
            f"'{' '.join(cmd)}' no respondió a tiempo (timeout de {timeout:.0f}s). "
            "¿Está corriendo ydotoold? (systemctl --user status ydotoold.service)"
        )


def type_text(text: str, key_delay_ms: int = 4) -> None:
    """Escribe el texto tecla por tecla en la ventana con foco.

    Ver advertencia arriba sobre el bug de caracteres acentuados. Usar
    solo cuando "paste" no sea viable.
    """
    if not text:
        return
    _run(["ydotool", "type", "--key-delay", str(key_delay_ms), "--", text])


def paste_text(text: str, terminal: bool = False) -> None:
    """Copia al portapapeles y simula el atajo de pegado.

    `terminal=True` usa Ctrl+Shift+V (la mayoría de terminales); de lo
    contrario usa Ctrl+V (la mayoría del resto de las apps).
    """
    if not text:
        return
    _copy_to_clipboard(text)
    keys = PASTE_KEYS_TERMINAL if terminal else PASTE_KEYS_NORMAL
    _run(["ydotool", "key", *keys])


def _copy_to_clipboard(text: str) -> None:
    session = os.environ.get("XDG_SESSION_TYPE", "")

    if session == "wayland" and shutil.which("wl-copy"):
        cmd = ["wl-copy"]
    elif shutil.which("xclip"):
        cmd = ["xclip", "-selection", "clipboard"]
    elif shutil.which("xsel"):
        cmd = ["xsel", "--clipboard", "--input"]
    elif shutil.which("wl-copy"):  # fallback si XDG_SESSION_TYPE no está seteado
        cmd = ["wl-copy"]
    else:
        raise InjectionError(
            "No hay herramienta de portapapeles disponible. Instalá "
            "wl-clipboard en Wayland (dnf install wl-clipboard), o "
            "xclip/xsel en X11."
        )

    # capture=False: ver el docstring de _run — esto es justo lo que
    # arregla el timeout de wl-copy.
    _run(cmd, input_text=text, capture=False, timeout=5)


class TextInjector:
    """Envoltorio con el modo configurado (paste por default)."""

    def __init__(self, mode: str = "paste"):
        if mode not in ("paste", "type"):
            raise ValueError(f"Modo de inyección inválido: {mode!r} (usar 'paste' o 'type')")
        self.mode = mode

    def inject(self, text: str, terminal: bool = False) -> None:
        if self.mode == "type":
            type_text(text)  # el flag `terminal` no aplica: tipear no usa Ctrl+V
        else:
            paste_text(text, terminal=terminal)
