"""Arregla `libcublas.so.12 is not found` / `libcudnn` faltante.

CTranslate2 (el motor que usa faster-whisper) viene compilado contra las
librerías **runtime** de CUDA 12 (cuBLAS/cuDNN/cuDART), sin importar qué
versión del CUDA Toolkit tengas instalada a nivel sistema — en tu caso 13.3,
que trae `libcublas.so.13`, no `.so.12`. Por eso `nvcc --version` puede andar
perfecto y aun así faster-whisper no encuentra la librería que busca: no es
que tu instalación de CUDA esté mal, es una versión que no coincide.

La solución es instalar los wheels de pip `nvidia-cublas-cu12` /
`nvidia-cudnn-cu12` / `nvidia-cuda-runtime-cu12` (son solo las librerías .so,
no un toolkit completo) y dejar que este módulo las cargue ANTES de que
CTranslate2 intente usarlas.

Cómo se cargan acá:
1. Los wheels de pip son namespace packages sin `__init__.py`: el módulo
   `nvidia.cublas.lib` no tiene `__file__` útil, así que las carpetas `lib`
   se resuelven con `importlib.util.find_spec`.
2. Setear `LD_LIBRARY_PATH` a mitad de proceso NO alcanza: glibc lo lee una
   sola vez al arrancar el proceso. Lo que sí funciona es **precargar** las
   librerías con `ctypes.CDLL` usando la ruta absoluta del wheel: una vez que
   `libcublas.so.12` está cargada en el proceso, cuando CTranslate2 pide esa
   misma soname, el loader la reutiliza.
3. Igual dejamos `LD_LIBRARY_PATH` actualizado por si algún subproceso
   necesitara las carpetas, pero la carga real la hace el preload.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys

# Paquetes de pip que traen librerías CUDA runtime como wheels sueltos.
_CUDA_LIB_MODULES = [
    "nvidia.cublas.lib",
    "nvidia.cudnn.lib",
    "nvidia.cuda_runtime.lib",
]


def _find_module_dirs() -> list[str]:
    """Devuelve las carpetas `lib` de los wheels nvidia-*-cu12 instalados.

    Los wheels son namespace packages (sin `__init__.py`), así que
    `module.__file__` es `None`; la ruta real está en
    `spec.submodule_search_locations`.
    """
    dirs: list[str] = []
    for module_name in _CUDA_LIB_MODULES:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            # find_spec() con un nombre con puntos (ej. "nvidia.cublas.lib")
            # necesita importar los paquetes padre para resolver su
            # __path__. Si el extra 'cuda' no está sincronizado, el
            # paquete 'nvidia' de más arriba ni siquiera existe, y en vez
            # de devolver None (como para un módulo top-level ausente),
            # find_spec directamente deja propagar el ModuleNotFoundError
            # del import fallido del padre. Tratamos eso igual que "no
            # está instalado".
            continue
        if spec is None or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            if loc not in dirs:
                dirs.append(loc)
    return dirs


def _preload_cuda_libs(lib_dirs: list[str]) -> list[str]:
    """Precarga por ruta absoluta todas las librerías .so de los wheels.

    CTranslate2 pide `libcublas.so.12` / `libcudnn.so.9` / `libcudart.so.12`.
    Cargarlas primero (ctypes) hace que, cuando CTranslate2 las pida por
    soname, el loader las reutilice — aunque `LD_LIBRARY_PATH` no las
    incluya.

    Se hace en pasadas iterativas porque las librerías dependen entre sí
    (libcudnn depende de libcublas, libcublas de libcudart, etc.) y el
    orden exacto varía según la versión del wheel.
    """
    pending: list[str] = []
    for lib_dir in lib_dirs:
        for entry in sorted(os.listdir(lib_dir)):
            # Sonames versionadas: libcublas.so.12, libcudnn.so.9, ...
            if entry.startswith("lib") and ".so." in entry:
                pending.append(os.path.join(lib_dir, entry))

    loaded: list[str] = []
    while pending:
        progressed = False
        still_pending: list[str] = []
        for path in pending:
            try:
                ctypes.CDLL(path)
                loaded.append(os.path.basename(path))
                progressed = True
            except OSError:
                still_pending.append(path)
        pending = still_pending
        if not progressed:
            break  # sin progreso: dependencia que nunca va a aparecer
    return loaded


def patch_ld_library_path() -> list[str]:
    """Precarga las librerías CUDA runtime de los wheels de pip y, además,
    deja `LD_LIBRARY_PATH` actualizado por si un subproceso lo necesita.

    Devuelve las carpetas de librerías encontradas (para loguear en el
    banner de arranque / status)."""

    if sys.platform != "linux":
        return []

    lib_dirs = _find_module_dirs()
    if not lib_dirs:
        return []

    # La parte que de verdad importa: dejar las librerías cargadas en el
    # proceso antes de que CTranslate2 las pida.
    _preload_cuda_libs(lib_dirs)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_parts = current.split(os.pathsep) if current else []
    new_parts = [p for p in lib_dirs if p not in current_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(new_parts + current_parts)

    return lib_dirs
