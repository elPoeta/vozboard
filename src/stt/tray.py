"""Ícono de bandeja del sistema (system tray) para el modo standalone.

Usa PySide6 (Qt) porque KDE Plasma es nativo Qt/StatusNotifierItem, así
que el ícono se integra sin dependencias extra de GTK/AppIndicator.

Nota para otros entornos: GNOME no muestra bandeja del sistema por
defecto desde hace años, sin importar qué toolkit uses — hace falta la
extensión "AppIndicator and KStatusNotifierItem Support". En KDE
funciona directo, sin nada que instalar.

## Sobre el menú de clic derecho que no mostraba "Salir"

Causa real (ya resuelta): `QMenu.addAction(action)` con un `QAction` ya
construido **no toma posesión del objeto** (a diferencia de
`addAction("texto")`, que sí crea la acción ya parentada al menú). Si no
se guarda una referencia Python al `QAction` — como pasaba acá con
`quit_action`, una variable local — el recolector de basura de Python lo
destruye apenas termina `__init__()`, y la acción desaparece del menú en
silencio, sin ningún error. La regla: todo `QAction`/`QMenu` construido
suelto tiene que guardarse como `self.algo`, no como variable local.

Aparte de eso, en algunos entornos (Qt instalado vía pip/wheel sin el
soporte de DBusMenu completo, ciertas combinaciones de Plasma + Qt) el
menú contextual que Qt intenta exportar vía StatusNotifierItem/DBusMenu
puede no renderizarse solo, aunque `setContextMenu()` esté bien puesto.
Por eso acá también escuchamos manualmente la señal `activated` con
razón `Context` y mostramos el menú a mano con `popup()` — es un
respaldo, no debería duplicar nada en los entornos donde el camino nativo
sí funciona (ahí ese código simplemente no se dispara).
"""

from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon


class Status(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()
    ERROR = auto()


_COLORS = {
    Status.IDLE: QColor("#7D8592"),
    Status.RECORDING: QColor("#E85D4C"),
    Status.PROCESSING: QColor("#E8A33D"),
    Status.ERROR: QColor("#E85D4C"),
}

_LABELS = {
    Status.IDLE: "inactivo — presioná el atajo para grabar",
    Status.RECORDING: "grabando…",
    Status.PROCESSING: "transcribiendo…",
    Status.ERROR: "error — ver terminal",
}


def _make_icon(color: QColor) -> QIcon:
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    margin = 8
    painter.drawEllipse(margin, margin, size - margin * 2, size - margin * 2)
    painter.end()

    return QIcon(pixmap)


class TrayApp(QObject):
    """Encapsula el ícono de bandeja y expone señales Qt (thread-safe:
    se pueden emitir desde otros hilos, Qt las encola al hilo principal)."""

    request_toggle = Signal()
    request_quit = Signal()
    request_set_paste_mode = Signal(bool)  # True = modo terminal (Ctrl+Shift+V)
    status_changed = Signal(object)  # Status

    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self._status = Status.IDLE
        self._default_terminal = False

        self.tray = QSystemTrayIcon()
        self.tray.setIcon(_make_icon(_COLORS[Status.IDLE]))

        self.menu = QMenu()

        self.toggle_action = QAction("Grabar / detener (modo por default)")
        self.toggle_action.triggered.connect(self.request_toggle.emit)
        self.menu.addAction(self.toggle_action)

        self.menu.addSeparator()

        paste_menu = self.menu.addMenu("Modo de pegado por default")
        self.action_paste_normal = QAction("Ctrl+V (apps normales)")
        self.action_paste_normal.setCheckable(True)
        self.action_paste_normal.setChecked(True)
        self.action_paste_terminal = QAction("Ctrl+Shift+V (terminales)")
        self.action_paste_terminal.setCheckable(True)

        self.action_paste_normal.triggered.connect(
            lambda: self.request_set_paste_mode.emit(False)
        )
        self.action_paste_terminal.triggered.connect(
            lambda: self.request_set_paste_mode.emit(True)
        )
        paste_menu.addAction(self.action_paste_normal)
        paste_menu.addAction(self.action_paste_terminal)

        self.menu.addSeparator()

        quit_action = self.quit_action = QAction("Salir")
        quit_action.triggered.connect(self.request_quit.emit)
        self.menu.addAction(quit_action)

        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_activated)

        self.status_changed.connect(self._update_icon)
        self._refresh_tooltip()
        self.tray.show()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.request_toggle.emit()
        elif reason == QSystemTrayIcon.ActivationReason.Context:
            # Respaldo manual — ver docstring del módulo.
            self.menu.popup(QCursor.pos())

    def _update_icon(self, status: Status) -> None:
        self._status = status
        self.tray.setIcon(_make_icon(_COLORS[status]))
        self._refresh_tooltip()

    def _refresh_tooltip(self) -> None:
        mode_label = "Ctrl+Shift+V (terminal)" if self._default_terminal else "Ctrl+V"
        self.tray.setToolTip(f"Vozboard — {_LABELS[self._status]} · pegado: {mode_label}")

    def set_status(self, status: Status) -> None:
        """Thread-safe: se puede llamar desde cualquier hilo."""
        self.status_changed.emit(status)

    def set_default_paste_mode(self, terminal: bool) -> None:
        """Actualiza los checkmarks del menú y el tooltip. Llamado desde
        el hilo principal (conectado a request_set_paste_mode, que solo
        se emite por interacción directa del usuario con el menú)."""
        self._default_terminal = terminal
        self.action_paste_normal.setChecked(not terminal)
        self.action_paste_terminal.setChecked(terminal)
        self._refresh_tooltip()

    def notify(self, title: str, message: str, is_error: bool = False) -> None:
        icon = (
            QSystemTrayIcon.MessageIcon.Critical
            if is_error
            else QSystemTrayIcon.MessageIcon.Information
        )
        self.tray.showMessage(title, message, icon, 4000)
