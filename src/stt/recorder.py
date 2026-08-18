"""Grabación desde micrófono para la CLI (no se usa en la web, que graba
directamente en el navegador)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000
CHANNELS = 1


class Recorder:
    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        if self.recording:
            self._chunks.append(indata.copy())

    def start(self) -> None:
        self._chunks = []
        self.recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self, audio_dir: str | Path) -> Path | None:
        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._stream = None

        if not self._chunks:
            return None

        audio = np.concatenate(self._chunks, axis=0)
        audio_dir = Path(audio_dir)
        audio_dir.mkdir(parents=True, exist_ok=True)

        filename = audio_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        write(str(filename), self.sample_rate, audio)
        return filename


def record_from_microphone(audio_dir: str | Path) -> Path | None:
    """Graba hasta que el usuario presiona ENTER, y devuelve la ruta del WAV."""

    recorder = Recorder()

    print()
    print("=" * 50)
    print("GRABACIÓN")
    print("=" * 50)
    print()
    print("Presiona ENTER para comenzar...")
    input()

    recorder.start()

    print()
    print("🎙️  Grabando...")
    print("Presiona ENTER para detener.")
    print()
    input()

    filename = recorder.stop(audio_dir)

    print()
    if filename is None:
        print("No se grabó ningún audio.")
        return None

    print("Audio guardado en:")
    print(filename)
    print()
    return filename
