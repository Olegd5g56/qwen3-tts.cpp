# qwen3-tts.cpp

> ⚠️ **Heads-up:** the changes in this fork were mostly written by Claude
> Code (Anthropic's AI coding agent). It runs and the upstream tests pass,
> but I can't promise production-grade robustness — read the diff and use at
> your own risk.

C++17 inference for [Qwen3-TTS](https://huggingface.co/collections/khimaros/qwen3-tts)
on [GGML](https://github.com/ggml-org/ggml). No Python at inference time.
OpenAI-compatible HTTP server with live streaming, a shared voice library,
and a one-shot CLI.

Languages: en, ru, zh, ja, ko, de, fr, es, it, pt.
Backends: CUDA / ROCm / Vulkan / Metal / CPU.

Measurements and design notes are in `docs/`: `optimization.md` (performance),
`known-issues.md` (bug log), `ggml-notes.md` (ggml side).

## Quickstart

```bash
git clone https://github.com/Olegd5g56/qwen3-tts.cpp.git
cd qwen3-tts.cpp && git submodule update --init --recursive

cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# models auto-download on first launch
./build/qwen3-tts-server \
    --hf-repo   khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF:Q8_0 \
    --hf-repo-v khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF:F16 \
    --host 0.0.0.0 --port 8080
```

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from qwen3-tts.cpp", "language": "en"}' \
  --output hello.wav
```

## Build

**Deps** (Arch names): `mpg123`, `libopusenc`, `lame`, a C++17 compiler,
CMake 3.14+. Vulkan SDK only for `-DGGML_VULKAN=ON`.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release                  # CPU
cmake -S . -B build -DGGML_CUDA=ON   -DCMAKE_BUILD_TYPE=Release # NVIDIA
cmake -S . -B build -DGGML_HIP=ON    -DCMAKE_BUILD_TYPE=Release # AMD ROCm
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release # Vulkan
```

Outputs: `build/qwen3-tts-server`, `build/qwen3-tts-cli`.

Prefer CUDA/ROCm/Metal over Vulkan on the same card.

- **CUDA**: set `-DCMAKE_CUDA_ARCHITECTURES=` to your card (75 Turing,
  86 Ampere, 89 Ada, 120 Blackwell). CUDA 13 needs GCC ≤ 15
  (`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`).
- **ROCm on gfx1030**: add `-DAMDGPU_TARGETS=gfx1030
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_PREFIX_PATH=/opt/rocm` and point
  the compilers at `/opt/rocm/llvm/bin/clang{,++}`.
- **Vulkan on AMD**: set `GGML_VK_DISABLE_MULTI_ADD=1` at runtime, or it is
  ~3x slower.

### Build toggles

| Option | Default | Effect |
|---|---|---|
| `QWEN3_TTS_NATIVE` | `OFF` | `-march=native` for this CPU (also pins ggml's `GGML_NATIVE`). Turn on only when the build host and the run host are the same machine — elsewhere the binary dies with `SIGILL`. `OFF` still assumes AVX2. |
| `QWEN3_TTS_SERVER` | `ON` | `OFF` skips building the server. |
| `QWEN3_TTS_TIMING` | `OFF` | Compiles in per-stage timing. |

## Docker

```bash
docker build -f Dockerfile.cuda   -t qwen3-tts:cuda   .   # Nvidia
docker build -f Dockerfile.vulkan -t qwen3-tts:vulkan .   # AMD/Intel
docker build -f Dockerfile.cpu    -t qwen3-tts:cpu    .   # CPU
```

Build args: `QWEN3_TTS_NATIVE=ON` (see the table above), and for
`Dockerfile.cuda` also `CUDA_ARCH=75` (default). `Dockerfile.cuda` carries a
`HEALTHCHECK`.

```bash
docker run --rm -it --gpus '"device=0"' \
    -p 8081:8081 \
    -v /path/to/models:/models:ro \
    -v /path/to/voices:/voices \
    -e TTS_MODEL=/models/talker.gguf \
    -e TTS_VOCODER=/models/vocoder.gguf \
    -e TTS_VOICES_DIR=/voices \
    -e TTS_HOST=0.0.0.0 \
    qwen3-tts:cuda
```

Set `-e TZ=<IANA name>` to match log timestamps to the host.

## Models

GGUFs: [`khimaros/qwen3-tts`](https://huggingface.co/collections/khimaros/qwen3-tts)
(F16 and Q8_0).

| Variant | Cloning | `instructions` | Built-in speakers |
|---------|---------|----------------|-------------------|
| 0.6B / 1.7B **Base** | yes | no | no |
| 0.6B / 1.7B **CustomVoice** | no | no | yes |
| 1.7B **VoiceDesign** | no | yes | no |
| **vocoder** (Qwen3-TTS-Tokenizer-12Hz) | — | — | — |

Cloning and the voice library need a **Base** model; loading anything else
disables the library at startup.

0.6B vs 1.7B: ~1 GB less VRAM, ~11% faster per frame, audibly worse quality.

**GPU memory** — peak on a long clip, everything resident:

| Model | Peak VRAM |
|---|---|
| 1.7B | ~3.2 GB |
| 0.6B | ~2.1 GB |

Budget for that plus anything else on the card; a CUDA out-of-memory aborts the
process. To cut it: `QWEN3_TTS_DECODE_BATCH=8` (−120 MB, −7% throughput) or
`QWEN3_TTS_LOW_MEM=1` (much lower peak, one model load per request).

**Auto-download**: `--hf-repo <repo>[:<quant>]` and `--hf-repo-v ...` (default
quant `Q8_0`, cached in `~/.cache/huggingface/`). **Local**: `-m talker.gguf
-v vocoder.gguf`; `-m` also takes a directory and finds the vocoder itself.

## Server

All flags have matching `TTS_*` env vars (CLI > env > default).

| Flag | Env | Default | What it does |
|------|-----|---------|--------------|
| `-m, --model` | `TTS_MODEL` | — | talker GGUF (file or directory) |
| `-v, --vocoder` | `TTS_VOCODER` | auto | vocoder GGUF |
| `--hf-repo` / `--hf-repo-v` | `TTS_HF_REPO` / `TTS_HF_REPO_V` | — | auto-download from HF |
| `--hf-file` / `--hf-file-v` | `TTS_HF_FILE` / `TTS_HF_FILE_V` | derived | override GGUF filename |
| `-H, --host` | `TTS_HOST` | `127.0.0.1` | listen address |
| `-p, --port` | `TTS_PORT` | `8080` | listen port |
| `-j, --threads` | `TTS_THREADS` | `4` | compute threads |
| `--voices-dir` | `TTS_VOICES_DIR` | — | voice library directory |
| `--idle-timeout` | `TTS_IDLE_TIMEOUT` | `0` | unload model after N idle seconds (0 = off); reloads lazily, voice library survives |
| `-V, --verbose` | `TTS_VERBOSE` | off | per-stage progress + timing logs |
| `--temperature`, `--top-k`, `--repetition-penalty`, `--seed` | `TTS_*` | `0.9` / `50` / `1.05` / `-1` | sampling defaults |

Notes:

- **One request at a time.** A second request waits for the first (tens of
  seconds on long text).
- `--temperature 0` degenerates on this model. Use a low temperature with a
  fixed `--seed` for repeatable output.
- The port opens before the voice library is warm. Voices encode on demand, so
  requests work immediately; the first call for a not-yet-warmed voice is
  slower.

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/health` | health check; answers while busy |
| `GET` | `/v1/models` | currently-loaded model id |
| `GET` | `/v1/audio/languages` | supported language codes |
| `GET` | `/v1/audio/voices` | list built-in + library voices |
| `POST` | `/v1/audio/voices` | upload an in-memory session voice (multipart) |
| `DELETE` | `/v1/audio/voices/:id` | drop a session voice |
| `POST` | `/v1/audio/speech` | synthesize (one-shot or streaming) |

`/health`:

```json
{"status":"ok","model_loaded":true,"busy":false,
 "voices":{"total":312,"warmed":312,"warming":false}}
```

`voices` is absent when no voice library is configured.

### `POST /v1/audio/speech`

```json
{
  "input": "Text to synthesize (max 4096 UTF-8 codepoints)",
  "voice": "default | <built-in name> | <library voice id>",
  "instructions": "(VoiceDesign only) describe the desired voice",
  "language": "en",
  "response_format": "mp3 (default, as in OpenAI) | wav | pcm | opus",
  "stream_format": "audio | sse",
  "stream_batch_size": 16,
  "temperature": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "seed": -1
}
```

`language` applies to the whole request — there is no per-clause markup. Set it
to the carrier language; Latin names and embedded foreign sentences ride along
fine.

- **Default** (no `stream_format`): full audio body once generation completes.
- **`stream_format=audio`**: HTTP chunked transfer. WAV uses a placeholder-size
  header so playback starts immediately; Opus is a self-contained Ogg stream;
  MP3 is self-framing.
- **`stream_format=sse`**: `speech.audio.delta` frames (base64) then
  `speech.audio.done`.

Live streaming needs **both** `stream_format` and a non-zero
`stream_batch_size`. PCM is 24 kHz mono S16LE. MP3 is LAME VBR `-V 4`
(~70 kbps mono, fixed — the OpenAI spec has no bitrate knob).

```bash
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello, streaming PCM straight to ALSA.",
       "response_format": "pcm", "stream_format": "audio",
       "stream_batch_size": 16, "language": "en"}' \
| aplay -q -f S16_LE -r 24000 -c 1
```

### Voice library

Two sources that never overlap.

**Disk-backed**, via `--voices-dir` — you manage it by editing files:

```
<voices-dir>/
  alice/
    sample.wav     # or sample.mp3 (wav wins if both exist)
    sample.txt     # optional transcript — enables ICL cloning
    cache.bin      # generated: embedding + reference codes
```

A voice is re-encoded (~1.4 s) whenever `sample.*` changes or the model variant
does. First use of an ICL voice also pays a ~2 s vocoder warm-up, cached
in-process afterwards. `DELETE` on a disk voice returns **409** — remove the
directory instead.

**Session**: `POST /v1/audio/voices` holds a voice in RAM. Survives idle
unload, dies on restart, **409** on a name that collides with a disk voice.

Both need a Base model.

## CLI

One-shot synthesis. Same env vars, same voice layout, same model rules. Output
format comes from the extension (`.wav`, `.mp3`, `.opus`/`.ogg`).

```bash
./build/qwen3-tts-cli -m ./models -t "Hello!" -o hello.wav
echo "From a pipe" | ./build/qwen3-tts-cli -m ./models -o piped.wav

# library voice / inline reference / built-in speaker
./build/qwen3-tts-cli -m ./base --voices-dir ./voices -v bob -t "Hi bob" -o bob.wav
./build/qwen3-tts-cli -m ./base -r reference.mp3 -t "Hello" -o cloned.wav
./build/qwen3-tts-cli -m ./customvoice -v alice -t "Hi" -o alice.wav

./build/qwen3-tts-cli -m ./model --voices-dir ./voices --list-voices
```

Voices load lazily — only the one passed to `-v` is encoded.

CLI-only flags: `-t/--text`, `-o/--output`, `-r/--reference`, `--ref-text`,
`-l/--language`, `-i/--instructions`, `--list-voices`, `--codebooks`,
`--streaming-batch-size`. Do not set `--codebooks` below 12 — it wrecks the
spoken text, not just the timbre.

## Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `QWEN3_TTS_PIPELINE` | auto | Overlap the vocoder with generation. Auto-on for CUDA/ROCm/Metal, off for Vulkan. `1`/`0` forces it. |
| `QWEN3_TTS_DECODE_BATCH` | 16 | Frames per vocoder batch. `8` saves ~120 MB VRAM and costs ~7%. |
| `QWEN3_TTS_FRAME_BUDGET` | on | Runaway guard: caps generated frames from input length. `0` disables it. |
| `QWEN3_TTS_LOW_MEM` | off | Never keep the talker and vocoder resident at once. Much lower peak VRAM, one model load per request, no overlap on the one-shot path. |
| `QWEN3_TTS_FORCE_CPU` | off | `1` keeps everything on the CPU. |
| `QWEN3_TTS_PROFILE_OPS` | off | Per-op timing table. Diagnostic only — disables fusion and syncs per node. |

## Testing

```bash
ctest --test-dir build
QWEN3_TTS_TEST_VOCODER=/path/to/tokenizer.gguf ctest --test-dir build
```

Tests without their fixtures report **Skipped**, so a red run is a real
regression. `QWEN3_TTS_TEST_MODEL` points at a talker GGUF;
`scripts/generate_deterministic_reference.py` regenerates the reference dumps.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) — Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) — Georgi Gerganov & contributors
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) — vocoder architecture
- Upstream chain:
  [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp)
  → [`khimaros/qwen3-tts.cpp`](https://github.com/khimaros/qwen3-tts.cpp)
  (1.7B support, ICL voice cloning, HTTP server, streaming vocoder, HF integration)
