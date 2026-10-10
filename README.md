# vozboard
Speech to text application from voice to input text area - wayland cuda/cpu whisper


Transcripción local con GPU (`faster-whisper` + CUDA), con tres formas
de usarla, todas sobre el mismo núcleo (`src/stt/core.py`):

- **CLI** (`stt`): grabar por micrófono o transcribir un archivo, desde la terminal.
- **Web** (`stt-web`): interfaz en el navegador con grabación en vivo (VU-meter),
  drag & drop de archivos, y descarga de `.txt` / `.srt` / `.vtt` / `.json`.
- **Standalone** (`stt-daemon`): ícono de bandeja + atajo de teclado global —
  hablás en cualquier campo de texto del sistema y el texto aparece solo,
  tipo VoiceInk/MacWhisper pero para Linux/Wayland. Ver sección 6.

## 1. Instalar con `uv`

```bash
uv python install 3.12
uv python pin 3.12
uv sync
uv run python --version
source .venv/bin/activate
sudo dnf install python3-devel
```

Esto crea el entorno virtual (`.venv/`) e instala todas las dependencias,
incluyendo `faster-whisper`, `fastapi` y `uvicorn`.

### ⚠️ Importante si vas a usar más de un extra (`cuda`, `daemon`)

`uv sync --extra X` **sincroniza** el entorno con exactamente lo que
pediste — no lo *suma* a lo que ya tenías instalado. Si corrés
`uv sync --extra cuda` y después `uv sync --extra daemon` por separado,
la segunda corrida te borra las dependencias de `cuda` (porque ese
comando, tal cual está escrito, le dice a uv "el entorno debe tener
exactamente los extras base + daemon", sin cuda). Y viceversa.

Si necesitás los dos (típicamente: GPU + modo standalone), pedilos
juntos en el mismo comando:

```bash
uv sync --extra cuda --extra daemon
```

o directamente todos los extras que tenga el proyecto:

```bash
uv sync --all-extras
```

También necesitás **FFmpeg** en el sistema (no lo instala `uv`):

```bash
sudo apt update
sudo apt install ffmpeg
```

### Nota sobre CUDA

Tener `nvcc` (CUDA Toolkit) no garantiza que `faster-whisper`/CTranslate2
encuentren las librerías de runtime que necesitan (cuBLAS/cuDNN). Esto es
especialmente cierto si tu CUDA Toolkit es más nuevo (por ejemplo 13.3):
CTranslate2 viene compilado contra las librerías **runtime** de CUDA 12,
así que aunque `nvcc --version` te muestre 13.3 perfecto, va a buscar
`libcublas.so.12` específicamente — y tu sistema solo tiene
`libcublas.so.13`. No es que tu GPU o tu instalación estén mal, es un
desalineamiento de versiones.

Si te aparece:

```text
Unable to load libcudnn_ops.so
Could not load library libcublas.so.12
```

instalá el extra `cuda` del proyecto — son wheels de pip con las
librerías CUDA 12 sueltas, no hace falta instalar nada a nivel sistema
ni tocar tu CUDA Toolkit 13.3:

```bash
uv sync --extra cuda
# si también vas a usar el modo standalone (sección 6), pedilo junto:
# uv sync --extra cuda --extra daemon
```

El proyecto detecta esos paquetes automáticamente y ajusta
`LD_LIBRARY_PATH` antes de cargar el modelo — no hace falta exportar
nada a mano. Vas a ver esto en el banner de arranque cuando funcione:

```text
LD_LIBRARY_PATH ajustado con librerías CUDA de pip:
  - /ruta/a/.venv/lib/.../nvidia/cublas/lib
  - /ruta/a/.venv/lib/.../nvidia/cudnn/lib
```

Si preferís, mientras tanto podés correr con `--device cpu` (más lento,
pero no depende de esto).

## 2. Uso — CLI

```bash
# Transcribir un archivo
uv run stt file audio.mp3

# Grabar desde el micrófono
uv run stt record

# Menú interactivo (estilo clásico)
uv run stt menu

# Opciones útiles
uv run stt file audio.mp3 --model large-v3 --device cuda --formats txt,srt,vtt,json
uv run stt file audio.mp3 --language en   # forzar idioma
uv run stt file audio.mp3 --language ""   # autodetectar idioma
```

Con tu **RTX 2000 Pro Blackwell de 8 GB**, `distil-large-v3` (el default)
es un buen punto de partida por velocidad; para máxima calidad usá
`--model large-v3`.

Los archivos de audio grabados quedan en `audio/`, y las transcripciones
en `output/`.

## 3. Uso — Web

```bash
uv run stt-web
```

Por defecto levanta en `http://0.0.0.0:8000`. Abrí `http://localhost:8000`
en el navegador.

Variables de entorno opcionales (antes de `uv run stt-web`):

```bash
export STT_MODEL=large-v3
export STT_DEVICE=cuda
export STT_COMPUTE_TYPE=float16
export STT_LANGUAGE=es
export STT_PORT=8000
```

El modelo se carga **una sola vez**, al arrancar el servidor, no en cada
request — así el primer request no paga el costo de carga.

### Cambiar idioma, modelo o dispositivo desde la web

La interfaz tiene tres selectores arriba, además de las variables de
entorno de arranque:

- **Idioma** — se aplica al toque, en cada transcripción, sin recargar
  nada. Es solo un parámetro que se manda junto con el audio.
- **Modelo** y **Dispositivo (CUDA/CPU)** — requieren recargar el modelo
  en memoria (puede tardar según el tamaño del modelo), así que tienen
  un botón **Aplicar** aparte. Mientras se recarga, las transcripciones
  quedan en espera (no se pierden, hacen cola) y se reanudan solas
  cuando termina.

Si preferís fijarlos de una vez al arrancar el server y no tocar nada
desde la web, las variables de entorno (`STT_MODEL`, `STT_DEVICE`, etc.)
siguen funcionando igual — los selectores simplemente parten de esos
valores.

La grabación desde el navegador usa `MediaRecorder` (requiere HTTPS o
`localhost`; si vas a exponerlo en red local con otro nombre de host,
vas a necesitar TLS para que el navegador permita el micrófono).

## 4. Dónde quedan los modelos y los archivos

Si en tu perfil de bash tenés definido tu entorno de IA:

```bash
# AI Environment
export AI_HOME="$HOME/AI"

export HF_HOME="$AI_HOME/models/huggingface"
export TORCH_HOME="$AI_HOME/models/torch"

export AI_MODELS="$AI_HOME/models"
export AI_VOICES="$AI_HOME/voices"
export AI_OUTPUT="$AI_HOME/output"
export AI_CACHE="$AI_HOME/cache"
```

el proyecto lo detecta solo y organiza todo ahí, sin que tengas que tocar
nada:

| Qué                          | Dónde queda                          |
|-------------------------------|---------------------------------------|
| Pesos del modelo Whisper       | `$AI_MODELS/whisper/`                 |
| Audio grabado/subido           | `$AI_OUTPUT/vozboard/audio/`    |
| Transcripciones (.txt/.srt/…)  | `$AI_OUTPUT/vozboard/transcripts/` |

`HF_HOME` no lo tocamos: como ya lo tenés exportado, `faster-whisper`
(vía `huggingface_hub`) lo respeta automáticamente para cualquier otro
asset que necesite (por ejemplo el modelo de VAD). El `download_root`
explícito de Whisper apunta aparte, a `$AI_MODELS/whisper`, para que los
pesos grandes de Whisper queden identificados en su propia carpeta y no
mezclados con el cache genérico de Hugging Face.

Si **no** tenés ese perfil cargado (por ejemplo corriendo esto en otra
máquina), no pasa nada: cae de vuelta a carpetas locales dentro del
proyecto (`./audio`, `./output`, `.cache/whisper`).

Podés confirmar qué rutas está usando con:

```bash
uv run stt file audio.mp3   # lo muestra en el banner de arranque
curl localhost:8000/api/status | python3 -m json.tool   # para la web
```

También podés pisar la carpeta de descarga puntualmente:

```bash
uv run stt file audio.mp3 --download-root /otra/ruta
STT_DOWNLOAD_ROOT=/otra/ruta uv run stt-web
```

## 6. Modo standalone (dictado global, tipo VoiceInk/MacWhisper)

Hay un tercer modo, además de CLI y web: un daemon con **ícono de
bandeja + atajo de teclado global**. Presionás el atajo en cualquier
lugar del sistema (un campo de texto, una terminal, lo que esté
enfocado), hablás, lo presionás de nuevo, y el texto transcripto
aparece pegado ahí mismo — sin abrir nada.

```bash
uv sync --extra daemon
# si también necesitás el extra cuda (GPU), pedilos juntos:
# uv sync --extra cuda --extra daemon
uv run stt-daemon
```

### Por qué esto es más difícil en Wayland que en X11

En X11 cualquier proceso puede simular teclas o "pegar" en la ventana
que esté enfocada (`xdotool`). Wayland lo bloquea a propósito por
seguridad: ninguna app puede inyectar eventos de teclado a otra sin
pasar por un mecanismo explícito del compositor. No hay forma de
evitar esto — es una decisión de diseño, no un bug.

La solución que usa este proyecto (y la que usan herramientas
equivalentes como `waystt`) es **`ydotool`**: en vez de hablarle al
compositor, crea un teclado virtual a nivel de kernel (`/dev/uinput`).
El compositor lo ve como un teclado físico más, así que funciona igual
en KWin, Mutter, Sway, o lo que sea — es la opción más agnóstica de
compositor que existe hoy. La contra es que necesita permisos a nivel
de kernel (grupo `input` o una regla `udev`), no son permisos de
aplicación normales.

Por la misma razón, el atajo de teclado global tampoco puede
registrarse con las APIs normales de la app — este proyecto usa
`evdev` para leer los eventos crudos del teclado directo del kernel
(sin "agarrar" el dispositivo, así que las teclas le siguen llegando
normalmente a la app enfocada). Esto también es agnóstico de
compositor/DE, a diferencia de atajos registrados vía KGlobalAccel
(específico de KDE) o el portal `GlobalShortcuts` (todavía dispar
entre compositores).

### ⚠️ Por qué el modo de inyección por default es "pegado", no "tipeo"

`ydotool type` simula cada tecla una por una, y tiene un bug conocido
(sin arreglar a la fecha) donde el texto **se corta apenas aparece un
carácter acentuado** — é, á, í, ó, ú, ñ. Para transcripción en español
esto no es un detalle menor, es perder la mitad del texto.

Por eso el modo por default es **`paste`**: copia el texto completo
(UTF-8, sin pasar por simulación de teclas) al portapapeles, y solo
simula el atajo Ctrl+V — dos códigos de tecla ASCII, ajenos al bug. Es
además más rápido para textos largos.

La contra de `paste` es que pisa tu portapapeles, y no funciona en
apps que interceptan Ctrl+V (algunos clientes KVM, ciertas VMs). Si te
pasa eso, podés forzar el modo tipeo con `STT_INJECT_MODE=type`, con
la salvedad del bug de acentos de arriba.

**Nota técnica sobre un bug ya corregido:** copiar al portapapeles vía
`wl-copy` se hace con `stdout`/`stderr` redirigidos a `/dev/null` en vez
de capturados. Si no, el proceso en segundo plano que `wl-copy` deja
corriendo (necesario para seguir sirviendo el contenido del portapapeles
mientras alguien no lo pida) hereda esos pipes abiertos, y todo el
pegado se cuelga hasta el timeout — este era el bug del "wl-copy
timeout" reportado inicialmente.

### Instalación paso a paso (Fedora + KDE)

```bash
# 1. ydotool + wl-clipboard (paquetes de sistema, no los instala uv)
sudo dnf install ydotool wl-clipboard

# 2. Dependencias Python del daemon
# (si también usás GPU, sumá --extra cuda acá: uv sync --extra cuda --extra daemon)
uv sync --extra daemon

# 3. Permisos para /dev/uinput sin correr nada como root
sudo cp packaging/60-uinput.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG input $USER
# cerrar sesión y volver a entrar acá, el grupo recién toma efecto al reloguear

# 4. ydotoold como servicio de usuario (arranca solo, sin sudo)
mkdir -p ~/.config/systemd/user
cp packaging/ydotoold.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ydotoold.service
systemctl --user status ydotoold.service   # confirmá que diga "active (running)"

# 5. Probar que la inyección funciona antes de meter el modelo en el medio
ydotool type "hola mundo"   # tenés que verlo escrito en donde tengas el cursor

# 6. Arrancar el daemon
uv run stt-daemon
```

Vas a ver un círculo gris en la bandeja del sistema. Hay **dos atajos**
por default, uno para pegado normal y otro para terminales (ver más
abajo por qué):

- **Ctrl derecho + Alt derecho** → graba/transcribe/pega con `Ctrl+V`
- **Ctrl izquierdo + Alt izquierdo** → graba/transcribe/pega con `Ctrl+Shift+V`

El círculo pasa a rojo mientras graba, a naranja mientras transcribe, y
el texto aparece pegado solo.

### Ctrl+V no funciona en terminales — por diseño, no es un bug

Casi ninguna terminal (Konsole, GNOME Terminal, etc.) usa `Ctrl+V` para
pegar — está libre o no hace nada — usan `Ctrl+Shift+V`. Como no hay
forma agnóstica de compositor de saber "¿la ventana con foco es una
terminal?" sin meterse con APIs específicas de KWin/Mutter (lo que
rompería el objetivo de que esto funcione en cualquier DE), la solución
es tener **dos atajos explícitos** — vos elegís cuál usar según dónde
estés parado. El ícono de bandeja también tiene un menú (clic derecho)
para fijar cuál de los dos es el "default" que se usa al hacer clic
directo en el ícono, y el tooltip del ícono siempre muestra cuál está
activo.

### Elegir tus propios atajos

Los defaults (`Ctrl der.+Alt der.` / `Ctrl izq.+Alt izq.`) se eligieron
por ser poco comunes, pero pueden chocar con tu configuración. Para
armar los tuyos:

```bash
uv run stt-daemon --list-keys
# presioná las teclas que querés usar, anotá los nombres que aparecen
```

y después:

```bash
export STT_HOTKEY="KEY_LEFTCTRL+KEY_LEFTALT+KEY_SPACE"            # pegado normal (Ctrl+V)
export STT_HOTKEY_TERMINAL="KEY_RIGHTCTRL+KEY_RIGHTALT+KEY_SPACE"  # pegado terminal (Ctrl+Shift+V)
uv run stt-daemon
```

o, sin exportar nada, directo por flag:

```bash
uv run stt-daemon --hotkey "KEY_LEFTCTRL+KEY_LEFTALT+KEY_SPACE" --hotkey-terminal "KEY_RIGHTCTRL+KEY_RIGHTALT+KEY_SPACE"
```

### Autostart al iniciar sesión

Dos formas, elegí una:

**Opción A — `.desktop` (recomendada, funciona en cualquier DE):**

```bash
mkdir -p ~/.config/autostart
cp packaging/vozboard-daemon.desktop ~/.config/autostart/
# editá el Exec= adentro si tu proyecto no está en ~/vozboard
```

**Opción B — servicio systemd de usuario (reinicio automático si se cae):**

```bash
mkdir -p ~/.config/systemd/user
cp packaging/stt-daemon.service ~/.config/systemd/user/
# editá WorkingDirectory= adentro
systemctl --user daemon-reload
systemctl --user enable --now stt-daemon.service
```

La opción A hereda el entorno completo de la sesión gráfica (más
confiable para que el ícono de bandeja y el atajo anden bien); la B da
reinicio automático pero a veces necesita ayuda extra con las variables
de entorno de sesión (`DISPLAY`, `WAYLAND_DISPLAY`) según tu distro.

### Flags de línea de comandos (además de las env vars)

```bash
uv run stt-daemon --device cpu --model medium --language en
uv run stt-daemon --inject-mode type   # forzar tipeo en vez de pegado
```

Igual que en `stt` y `stt-web`, cada flag pisa la env var equivalente:

```bash
export STT_MODEL=distil-large-v3      # o large-v3, etc.
export STT_DEVICE=cuda                # compute_type se auto-ajusta a int8 en cpu, float16 en cuda
export STT_LANGUAGE=es
export STT_HOTKEY="KEY_RIGHTCTRL+KEY_RIGHTALT"
export STT_HOTKEY_TERMINAL="KEY_LEFTCTRL+KEY_LEFTALT"
export STT_INJECT_MODE=paste          # "paste" (default) o "type"
```

### Troubleshooting

| Síntoma | Causa probable |
|---|---|
| Ningún atajo dispara nada | `sudo usermod -aG input $USER` y reiniciá sesión; confirmá con `groups` que aparece `input` |
| No aparece nada pegado, o tarda ~30s y tira error de timeout | Corregido — si te sigue pasando: `systemctl --user status ydotoold.service`, si no está "active" `ydotool`/`wl-copy` no tienen a quién hablarle |
| Se corta el texto en la primera tilde | Estás en modo `type`; volvé a `STT_INJECT_MODE=paste` (default) |
| No pasa nada en la terminal con el atajo de `Ctrl+V` | Usá el otro atajo (el de `Ctrl izq.+Alt izq.` por default), pensado justo para eso — o cambiá el default desde el menú del ícono |
| No aparece el menú al hacer clic derecho en el ícono | Ya debería andar (hay un respaldo manual); si seguís sin verlo, probá clic izquierdo primero para confirmar que el ícono responde, y contame qué versión de Qt/Plasma tenés |
| No aparece el ícono en la bandeja (GNOME) | Instalá la extensión "AppIndicator and KStatusNotifierItem Support" — en KDE no hace falta nada |
| `stt-daemon --device cpu` no cambiaba nada | Corregido — antes el daemon no leía flags, solo env vars |
| Funciona con `ydotool type "test"` a mano pero no desde el daemon | Confirmá que `ydotoold` siga corriendo (`systemctl --user status`), y que no haya otro `YDOTOOL_SOCKET` distinto seteado en el entorno del daemon |

### Qué falta / limitaciones conocidas

- No hay traducción de layout de teclado (AZERTY, etc.) para el modo
  `type` — no debería importar mientras uses el modo `paste` (default).
- El atajo global no distingue "sostener para grabar" (push-to-talk) de
  "un toque para alternar" — es siempre toggle. Push-to-talk real
  requeriría medir tiempo de retención, queda como mejora futura.
- Sin detección de reconexión de teclado en caliente: si conectás un
  teclado nuevo después de arrancar el daemon, hay que reiniciarlo.
- El modo `paste` pisa tu portapapeles (guarda lo que tenías antes ahí
  si te importa, antes de dictar).

## 7. Estructura

```
vozboard/
├── pyproject.toml
├── src/stt/
│   ├── core.py       # carga de modelo, transcripción, export txt/srt/vtt/json
│   ├── cuda_env.py    # fix de libcublas/libcudnn faltantes (CUDA Toolkit != runtime CTranslate2)
│   ├── paths.py       # resuelve rutas a partir de AI_HOME/AI_MODELS/AI_OUTPUT/AI_CACHE
│   ├── recorder.py   # grabación por micrófono (CLI y daemon)
│   ├── cli.py         # entry point `stt`
│   ├── web.py         # entry point `stt-web` (FastAPI)
│   ├── hotkey.py       # atajo global vía evdev (agnóstico X11/Wayland/DE)
│   ├── inject.py       # escribir/pegar texto en la ventana con foco vía ydotool
│   ├── tray.py          # ícono de bandeja (PySide6/Qt)
│   └── daemon.py        # entry point `stt-daemon`, conecta todo lo anterior
├── web/
│   └── index.html     # interfaz web (sin build step, HTML+CSS+JS plano)
├── packaging/
│   ├── 60-uinput.rules             # regla udev para /dev/uinput sin sudo
│   ├── ydotoold.service            # systemd --user para ydotoold
│   ├── stt-daemon.service          # systemd --user para stt-daemon (opcional)
│   └── vozboard-daemon.desktop     # autostart XDG (recomendado)
├── audio/              # audios grabados/subidos
└── output/             # transcripciones generadas
```

## Próximos pasos posibles

- Streaming de transcripción mientras se graba (en vez de esperar a parar).
- Historial persistente de transcripciones (SQLite) en la web.
- Diarización de hablantes.


uv sync --all-extras
