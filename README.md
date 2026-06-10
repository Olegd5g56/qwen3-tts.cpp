# qwen3-tts.cpp

> ⚠️ **Heads-up:** the changes in this fork were mostly written by Claude
> Code (Anthropic's AI coding agent). It runs and the upstream tests pass,
> but I can't promise production-grade robustness — read the diff and use at
> your own risk.

C++17 inference for [Qwen3-TTS](https://huggingface.co/collections/khimaros/qwen3-tts)
on [GGML](https://github.com/ggml-org/ggml). No Python at inference time.
Ships an OpenAI-compatible HTTP server with live streaming, a shared voice
library, and a one-shot CLI.

## What it does

- `POST /v1/audio/speech` — OpenAI-compatible, output as `wav` / `pcm` / `opus` / `mp3`
- Live streaming: chunks flushed as the vocoder produces them (chunked transfer or SSE)
- Voice library shared by server + CLI — drop reference audio in a directory,
  or upload session voices to the server in RAM
- Voice cloning via Mimi-codec ICL prefix (Base models), built-in presets
  (CustomVoice), or free-form `instructions` (VoiceDesign)
- Languages: en, ru, zh, ja, ko, de, fr, es, it, pt
- Backends: Vulkan / CUDA / Metal / HIP / CPU (runtime selection via GGML)
- Idle-unload watchdog for memory-constrained hosts

## Quickstart

```bash
git clone https://github.com/Olegd5g56/qwen3-tts.cpp.git
cd qwen3-tts.cpp && git submodule update --init --recursive

cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
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

**Deps** (Arch package names; equivalents on other distros): `mpg123`,
`libopusenc`, `lame`, a C++17 compiler, and CMake 3.14+. Add the Vulkan SDK
if you enable `-DGGML_VULKAN=ON`.

**Backends** — pick one:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release                  # CPU
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release # Vulkan
cmake -S . -B build -DGGML_HIP=ON    -DCMAKE_BUILD_TYPE=Release # AMD ROCm
cmake -S . -B build -DGGML_CUDA=ON   -DCMAKE_BUILD_TYPE=Release # NVIDIA
```

Outputs: `build/qwen3-tts-server` and `build/qwen3-tts-cli`.
Toggles: `QWEN3_TTS_SERVER=OFF` skips the server,
`QWEN3_TTS_TIMING=ON` compiles in per-stage timing.

## Docker

Two backend-specific Dockerfiles. The binary is built with `-march=native`,
so the image is tuned to whatever CPU ran `docker build` — rebuild if you
move it to a different host.

```bash
git submodule update --init --recursive

docker build -f Dockerfile.vulkan -t qwen3-tts:vulkan .   # AMD/Intel/NV via host Vulkan
docker build -f Dockerfile.cpu    -t qwen3-tts:cpu    .   # CPU only
```

Models and voices are mounted, never baked in:

```bash
docker run --rm -it \
    --device /dev/dri --group-add video \
    -p 8081:8081 \
    -v /path/to/models:/models:ro \
    -v /path/to/voices:/voices:ro \
    -e TZ=Europe/Warsaw \
    -e TTS_MODEL=/models/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
    -e TTS_VOCODER=/models/Qwen3-TTS-Tokenizer-12Hz-F16.gguf \
    -e TTS_VOICES_DIR=/voices \
    qwen3-tts:vulkan
```

CPU image: drop `--device /dev/dri --group-add video`. Container time
defaults to UTC; set `-e TZ=...` if you want logs in local time. Any local
`docker-compose.yml` can reference the tag — no `build:` section needed.

## Models

GGUFs live in the [`khimaros/qwen3-tts`](https://huggingface.co/collections/khimaros/qwen3-tts)
collection (F16 and Q8_0).

| Variant | Cloning | `instructions` | Built-in speakers |
|---------|---------|----------------|-------------------|
| 0.6B / 1.7B **Base** | yes | no | no |
| 0.6B / 1.7B **CustomVoice** | no | no | yes |
| 1.7B **VoiceDesign** | no | yes | no |
| shared **vocoder** (Qwen3-TTS-Tokenizer-12Hz) | — | — | — |

Only **Base** has a speaker encoder, so voice cloning (file or library)
only works there. CustomVoice ships built-in presets addressed by name;
VoiceDesign takes free-form `instructions`. Loading a non-Base model
disables the voice library at startup.

**Auto-download** via `--hf-repo <repo>[:<quant>]` and `--hf-repo-v ...`
(default quant `Q8_0`, cached under `~/.cache/huggingface/`).
**Local files**: `-m /path/to/talker.gguf -v /path/to/vocoder.gguf`. `-m`
also accepts a directory — the vocoder is auto-discovered.

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
| `--idle-timeout` | `TTS_IDLE_TIMEOUT` | `0` | unload model after N idle seconds (0 = off) |
| `-V, --verbose` | `TTS_VERBOSE` | off | per-stage progress + timing logs |
| `--temperature`, `--top-k`, `--repetition-penalty`, `--seed` | `TTS_*` | `0.9` / `50` / `1.05` / `-1` | sampling defaults |

### Endpoints

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/health` | health check |
| `GET` | `/v1/models` | currently-loaded model id |
| `GET` | `/v1/audio/languages` | supported language codes |
| `GET` | `/v1/audio/voices` | list built-in + library voices |
| `POST` | `/v1/audio/voices` | upload an in-memory session voice (multipart) |
| `DELETE` | `/v1/audio/voices/:id` | drop a session voice |
| `POST` | `/v1/audio/speech` | synthesize (one-shot or streaming) |

### `POST /v1/audio/speech`

```json
{
  "input": "Text to synthesize (max 4096 UTF-8 codepoints)",
  "voice": "default | <built-in name> | <library voice id>",
  "instructions": "(VoiceDesign only) describe the desired voice",
  "language": "en",
  "response_format": "wav | pcm | opus | mp3",
  "stream_format": "audio | sse",
  "stream_batch_size": 16,
  "temperature": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "seed": -1
}
```

- **Default** (no `stream_format`): full audio body after generation completes.
- **`stream_format=audio`**: HTTP chunked transfer in the chosen
  `response_format`. WAV uses a placeholder-size header so playback starts
  immediately; Opus produces a self-contained Ogg stream; MP3 is self-framing.
- **`stream_format=sse`**: Server-Sent Events with `speech.audio.delta`
  frames (base64 audio) and a final `speech.audio.done` event.

For *live* streaming you need **both** `stream_format` **and** a non-zero
`stream_batch_size`. PCM is always 24 kHz mono S16LE. MP3 is LAME VBR `-V 4`
speech-tuned (~70 kbps mono, fixed — OpenAI's spec has no bitrate knob and we
mirror it).

### Streaming example

```bash
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello, streaming PCM straight to ALSA.",
    "response_format": "pcm",
    "stream_format": "audio",
    "stream_batch_size": 16,
    "language": "en"
  }' \
| aplay -q -f S16_LE -r 24000 -c 1
```

### Voice library

Two sources that never overlap:

**1. Disk-backed (curated).** `--voices-dir <path>` is read-only from the
server's perspective — you manage it by editing files.

```
<voices-dir>/
  alice/
    sample.wav     # or sample.mp3 (wav wins if both exist)
    sample.txt     # optional reference transcript — enables ICL
    cache.bin      # auto-regenerated embedding + ref_codes
  bob/
    sample.mp3
    cache.bin
```

The server pre-warms the whole library at startup. Cache invalidation is by
file mtime, so editing the sample re-encodes on next use.
`DELETE /v1/audio/voices/:id` returns **409** on a disk-backed id — delete
the directory instead.

For ICL voices the vocoder warm-up (decoding the reference frames so output
starts in the reference timbre, ~5 s) runs once per voice and is then cached
in-process. Repeat requests with the same voice skip it entirely — streaming
time-to-first-audio drops from ~7 s to ~1.5 s. The cache survives idle
unload; it resets on process restart.

**2. Session (ephemeral).** `POST /v1/audio/voices` decodes the upload and
holds it in RAM. Survives idle-unload, dies on process restart. Name
collisions with a disk voice return **409**.

Voice cloning needs the Base variant. On CustomVoice / VoiceDesign the
directory is ignored and uploads are rejected.

### Idle unload

`--idle-timeout 600` drops the loaded model after 10 minutes of inactivity
and reloads it lazily on the next request. Handy for hot-swapping variants
through something like `llama-swap`. Voice library state survives reloads.

## CLI

`qwen3-tts-cli` is a one-shot synthesis tool. Same env vars, same voice
layout, same model-type rules. Output format is picked from the file
extension (`.wav`, `.mp3`, `.opus` / `.ogg`).

```bash
./build/qwen3-tts-cli -m ./models -t "Hello!" -o hello.wav
./build/qwen3-tts-cli -m ./models -t "Hello!" -o hello.mp3
echo "From a pipe" | ./build/qwen3-tts-cli -m ./models -o piped.wav

# library voice (Base)
./build/qwen3-tts-cli -m ./base --voices-dir ./voices -v bob \
    -t "Hello, bob" -o bob.wav

# inline reference audio (Base)
./build/qwen3-tts-cli -m ./base -r reference.mp3 -t "Hello" -o cloned.wav

# built-in speaker (CustomVoice)
./build/qwen3-tts-cli -m ./customvoice -v alice -t "Hi" -o alice.wav

# list voices without encoding anything
./build/qwen3-tts-cli -m ./model --voices-dir ./voices --list-voices
```

The CLI loads voices lazily — only the one passed to `-v` is encoded.
Cache miss prints `voice 'X': loading... / ready (N.Ns)`; warm cache is
sub-second.

CLI-only flags on top of the server table: `-t/--text`, `-o/--output`,
`-r/--reference` (inline audio), `--ref-text` (transcript for `-r`),
`-l/--language`, `-i/--instructions`, `--list-voices`,
`--streaming-batch-size`.

## Architecture

```
Text ──► [Tokenizer] ──► token IDs ─┐
                                    │
Reference WAV ──► [Speaker Encoder] ──► 1024-dim x-vector
Reference WAV ──► [Audio Tokenizer Enc] ──► ICL prefix codes
                                    │
        prefix + token IDs ──► [TTS Transformer] ──► speech codes
                                    │
                                    ▼
                              [Vocoder] ──► 24 kHz PCM
```

Each frame is produced in two stages: the **Talker** emits a hidden state
and codebook-0 logits, then a 5-layer **Code Predictor** autoregressively
emits codebooks 1–15. The vocoder decodes those codes into 24 kHz mono.
Under streaming, the vocoder runs on rolling chunks with preserved KV/tail
state, so audio flows out while the transformer is still generating.

## Testing

```bash
bash scripts/run_all_tests.sh              # full suite
uv run python scripts/compare_e2e.py       # Python ↔ C++ parity
```

Per-stage executables live in `build/test_*` — see the script for invocation.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) — Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) — Georgi Gerganov & contributors
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) — vocoder architecture
- Upstream chain:
  [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp)
  → [`khimaros/qwen3-tts.cpp`](https://github.com/khimaros/qwen3-tts.cpp)
  (1.7B support, ICL voice cloning, HTTP server, streaming vocoder, HF integration)
