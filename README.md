# qwen3-tts.cpp

C++17 inference for [Qwen3-TTS](https://huggingface.co/collections/khimaros/qwen3-tts)
built on [GGML](https://github.com/ggml-org/ggml). Ships an OpenAI-compatible
HTTP server with live audio streaming, voice cloning, and a CLI for one-shot
synthesis. No Python or PyTorch at inference time.

## Features

- OpenAI-compatible `/v1/audio/speech` server with `wav`, `pcm`, and `opus` output
- Live streaming: PCM/WAV/Opus chunks flushed to the wire as the vocoder produces
  them (`stream_format=audio` or SSE)
- Voice library shared between server and CLI — drop reference WAVs into a
  directory and address them by name
- Voice cloning via Mimi-codec ICL prefix (Base models) and built-in speaker
  presets (CustomVoice models)
- Voice steering via `instructions` (VoiceDesign 1.7B)
- Multi-language: `en`, `ru`, `zh`, `ja`, `ko`, `de`, `fr`, `es`, `it`, `pt`
- Backend selection at runtime: Vulkan / CUDA / Metal / CPU (via GGML)
- Idle-unload watchdog for memory-constrained hosts
- F16 and Q8_0 GGUFs auto-downloaded from Hugging Face

## Quickstart

The shortest path to a running server (Vulkan + CPU fallback, Arch Linux):

```bash
git clone https://github.com/Olegd5g56/qwen3-tts.cpp.git
cd qwen3-tts.cpp
git submodule update --init --recursive

# build with Vulkan
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# run — models are downloaded on first launch
./build/qwen3-tts-server \
    --hf-repo   khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF:Q8_0 \
    --hf-repo-v khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF:F16 \
    --host 0.0.0.0 --port 8080
```

Synthesize:

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Hello from qwen3-tts.cpp", "language": "en"}' \
  --output hello.wav
```

## Build

### Dependencies

- C++17 compiler (GCC 11+ or Clang 14+)
- CMake 3.14+
- `libopusenc` (server, Ogg/Opus encoding) — Arch: `pacman -S libopusenc`
- Vulkan SDK if `-DGGML_VULKAN=ON`

GGML is vendored as a submodule and built as part of the top-level CMake.
Backend selection is passed through to GGML:

```bash
# CPU only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Vulkan (broad GPU support, recommended for AMD/Intel/Nvidia)
cmake -S . -B build -DGGML_VULKAN=ON   -DCMAKE_BUILD_TYPE=Release

# HIP (AMD ROCm)
cmake -S . -B build -DGGML_HIP=ON      -DCMAKE_BUILD_TYPE=Release

# CUDA
cmake -S . -B build -DGGML_CUDA=ON     -DCMAKE_BUILD_TYPE=Release
```

Build options:

- `QWEN3_TTS_SERVER=ON/OFF` — build the HTTP server (default: ON)
- `QWEN3_TTS_TIMING=ON/OFF` — compile-in detailed per-stage timing (default: OFF)

Outputs:

- `build/qwen3-tts-server` — HTTP server
- `build/qwen3-tts-cli`    — one-shot CLI

## Models

Pre-converted GGUFs live in the [`khimaros/qwen3-tts`](https://huggingface.co/collections/khimaros/qwen3-tts)
collection. Each repo ships F16 and Q8_0 quants.

| Repo | Variant | Voice cloning | `instructions` | Built-in speakers |
|------|---------|---------------|----------------|-------------------|
| [`khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-0.6B-Base-GGUF) | 0.6B Base | yes | no | no |
| [`khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF) | 1.7B Base | yes | no | no |
| [`khimaros/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-0.6B-CustomVoice-GGUF) | 0.6B CustomVoice | no | no | yes |
| [`khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF) | 1.7B CustomVoice | no | no | yes |
| [`khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF) | 1.7B VoiceDesign | no | yes | no |
| [`khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF`](https://huggingface.co/khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF) | shared vocoder | — | — | — |

Only **Base** variants support voice cloning (they have a speaker encoder).
CustomVoice variants use built-in speaker presets addressed by name.
VoiceDesign uses free-form `instructions` to describe the desired voice.
The CLI and server both warn and disable the voice library if a non-Base
model is loaded.

### Auto-download via `--hf-repo`

```bash
./build/qwen3-tts-server \
    --hf-repo   khimaros/Qwen3-TTS-12Hz-1.7B-Base-GGUF:Q8_0 \
    --hf-repo-v khimaros/Qwen3-TTS-Tokenizer-12Hz-GGUF:F16
```

Format is `<repo>[:<quant>]` (default `Q8_0`). Override the exact filename
with `--hf-file` / `--hf-file-v`. Downloaded files are cached under
`~/.cache/huggingface/`.

### Manual local files

```bash
./build/qwen3-tts-server \
    -m /path/to/qwen3-tts-base.gguf \
    -v /path/to/qwen3-tts-tokenizer.gguf
```

`-m` accepts either a single GGUF file or a directory containing both the
talker and the tokenizer; in the directory case the vocoder is auto-discovered.

## Server

### Configuration

All flags have matching `TTS_*` environment variables (CLI flags override env,
env overrides defaults). Convenient for Docker / systemd.

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `-m, --model <path>` | `TTS_MODEL` | — | talker GGUF (file or directory) |
| `-v, --vocoder <file>` | `TTS_VOCODER` | auto | vocoder GGUF |
| `-hf, --hf-repo <repo[:quant]>` | `TTS_HF_REPO` | — | HuggingFace talker repo |
| `--hf-file <name>` | `TTS_HF_FILE` | derived | override talker GGUF filename |
| `--hf-repo-v <repo[:quant]>` | `TTS_HF_REPO_V` | — | HuggingFace vocoder repo |
| `--hf-file-v <name>` | `TTS_HF_FILE_V` | derived | override vocoder GGUF filename |
| `-H, --host <host>` | `TTS_HOST` | `127.0.0.1` | listen address |
| `-p, --port <port>` | `TTS_PORT` | `8080` | listen port |
| `-j, --threads <n>` | `TTS_THREADS` | `4` | compute threads |
| `--voices-dir <dir>` | `TTS_VOICES_DIR` | — | voice library directory |
| `--idle-timeout <s>` | `TTS_IDLE_TIMEOUT` | `0` | unload model after N idle seconds (0 = off) |
| `-V, --verbose` | `TTS_VERBOSE` | off | per-stage progress and timing |
| `--temperature <f>` | `TTS_TEMPERATURE` | `0.9` | default sampling temperature |
| `--top-k <n>` | `TTS_TOP_K` | `50` | default top-k |
| `--repetition-penalty <f>` | `TTS_REPETITION_PENALTY` | `1.05` | default repetition penalty |
| `--seed <n>` | `TTS_SEED` | `-1` | default sampling seed (-1 = random) |

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | health check |
| `GET` | `/v1/models` | currently-loaded model id |
| `GET` | `/v1/audio/languages` | supported language codes |
| `GET` | `/v1/audio/voices` | list built-in + library voices |
| `POST` | `/v1/audio/voices` | create a voice from reference audio (multipart) |
| `DELETE` | `/v1/audio/voices/:id` | delete a library voice |
| `POST` | `/v1/audio/speech` | synthesize (one-shot or streaming) |

### `POST /v1/audio/speech`

```json
{
  "input": "Text to synthesize (max 4096 chars)",
  "voice": "default | <built-in name> | <library voice id>",
  "instructions": "(VoiceDesign only) describe the desired voice",
  "language": "en",
  "response_format": "wav | pcm | opus",
  "stream_format": "audio | sse",
  "stream_batch_size": 16,
  "temperature": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "seed": -1
}
```

Output:

- **One-shot** (default, `stream_format` omitted): the full audio in
  `response_format` after generation completes.
- **`stream_format=audio`**: HTTP chunked transfer with bytes in the chosen
  `response_format`. WAV uses a placeholder-size header so playback can begin
  immediately; Opus produces a self-contained Ogg stream.
- **`stream_format=sse`**: Server-Sent Events emitting `speech.audio.delta`
  frames (base64-encoded audio) followed by `speech.audio.done` with usage
  and timing.

For *live* streaming you must set **both** `stream_format` and a non-zero
`stream_batch_size`. With `stream_batch_size=0` the whole utterance is
buffered before being flushed.

### Streaming example

Pipe straight into ALSA for the lowest latency:

```bash
curl -sN -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Hello, this is streaming PCM straight to the speakers.",
    "response_format": "pcm",
    "stream_format": "audio",
    "stream_batch_size": 16,
    "language": "en"
  }' \
| aplay -q -f S16_LE -r 24000 -c 1
```

PCM is always 24 kHz, mono, signed 16-bit little-endian.

### Voice library

Pass `--voices-dir <path>` to the server. The directory layout is:

```
<voices-dir>/
  alice/
    sample.wav
    sample.txt     # optional reference transcript (enables ICL)
    .cache.bin     # encoded embedding + ref_codes, regenerated automatically
  bob/
    sample.wav
    .cache.bin
```

The cache is invalidated by file mtime, so editing `sample.wav` or `sample.txt`
triggers a re-encode on the next request that uses the voice. Voices added via
`POST /v1/audio/voices` are written here.

Voice library entries are only loaded on Base models. With CustomVoice or
VoiceDesign models the directory is ignored (with a warning at startup),
because their architecture has no speaker encoder.

### Idle unload

`--idle-timeout 600` drops the loaded model after 10 minutes of inactivity
and reloads it lazily on the next request. Useful when running several model
variants on the same host through something like `llama-swap`. Voice library
membership is preserved across reloads (it's stored on disk).

## CLI

`qwen3-tts-cli` is a one-shot synthesis tool. Same `TTS_*` env vars as the
server where they overlap, same voice-library layout, same model-type rules.

```bash
# basic
./build/qwen3-tts-cli -m ./models -t "Hello!" -o hello.wav

# from stdin
echo "Hello from a pipe" | ./build/qwen3-tts-cli -m ./models -o piped.wav

# built-in speaker (CustomVoice model)
./build/qwen3-tts-cli -m ./customvoice -v alice -t "Hi there" -o alice.wav

# library voice (Base model)
./build/qwen3-tts-cli -m ./base --voices-dir ./voices -v bob \
    -t "Hello, bob" -o bob.wav

# inline reference WAV (Base model)
./build/qwen3-tts-cli -m ./base -r reference.wav -t "Hello" -o cloned.wav

# list everything visible to the loaded model
./build/qwen3-tts-cli -m ./model --voices-dir ./voices --list-voices
```

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `-m, --model <path>` | `TTS_MODEL` | — | talker GGUF (file or directory) |
| `--vocoder <file>` | `TTS_VOCODER` | auto | vocoder GGUF |
| `-t, --text <s>` | — | stdin | text to synthesize (or piped via stdin) |
| `-o, --output <file>` | — | `output.wav` | output WAV path |
| `-r, --reference <file>` | — | — | inline reference WAV (mutually exclusive with `-v`) |
| `-v, --voice <name>` | `TTS_VOICE` | — | built-in or library voice |
| `--voices-dir <dir>` | `TTS_VOICES_DIR` | — | voice library directory |
| `--list-voices` | — | — | print available voices and exit |
| `--ref-text <s>` | — | — | reference transcript for `-r` (enables ICL) |
| `-l, --language <code>` | `TTS_LANGUAGE` | `en` | one of: en, ru, zh, ja, ko, de, fr, es, it, pt |
| `-i, --instructions <s>` | — | — | voice steering (VoiceDesign 1.7B only) |
| `--temperature <f>` | `TTS_TEMPERATURE` | `0.9` | sampling temperature (0 = greedy) |
| `--top-k <n>` | `TTS_TOP_K` | `50` | top-k sampling |
| `--top-p <f>` | — | `1.0` | reserved (not yet wired into sampling) |
| `--max-tokens <n>` | — | `2048` | maximum audio tokens to generate |
| `--repetition-penalty <f>` | `TTS_REPETITION_PENALTY` | `1.05` | repetition penalty |
| `--seed <n>` | `TTS_SEED` | `-1` | sampling seed |
| `--streaming-batch-size <n>` | — | `0` | enable streaming vocoder decode (parity-tested against one-shot) |
| `-j, --threads <n>` | `TTS_THREADS` | `4` | compute threads |

## Architecture

```
Text ──► [Tokenizer] ──► token IDs ─┐
                                    │
Reference WAV ──► [Speaker Encoder] ──► speaker embedding (1024-dim x-vector)
Reference WAV ──► [Audio Tokenizer Enc] ──► ref_codes (ICL prefix)
                                    │
        prefix + token IDs ──► [TTS Transformer] ──► speech codes
                                    │
                                    ▼
                              [Vocoder] ──► 24 kHz PCM
```

The TTS transformer generates speech codes in two stages per frame:

1. **Talker** (28 layers for 0.6B / larger for 1.7B) produces a hidden state and codebook-0 logits.
2. **Code Predictor** (5 layers) autoregressively emits codebooks 1–15 from that hidden state.

The vocoder decodes these codes into a 24 kHz waveform. Under streaming mode
the vocoder is run on rolling chunks with preserved KV/tail state, so PCM is
produced frame-by-frame while the transformer is still generating.

### Source layout

| File | Component |
|------|-----------|
| `src/text_tokenizer.{h,cpp}` | BPE text tokenizer (loaded from GGUF) |
| `src/audio_tokenizer_encoder.{h,cpp}` | ECAPA-TDNN x-vector + Mimi codec encoder |
| `src/audio_tokenizer_decoder.{h,cpp}` | WavTokenizer vocoder (with streaming state) |
| `src/tts_transformer.{h,cpp}` | Talker + code predictor |
| `src/qwen3_tts.{h,cpp}` | Pipeline orchestration |
| `src/voice_store.{h,cpp}` | Shared on-disk voice library |
| `src/main.cpp` | CLI |
| `src/server.cpp` | HTTP server |

## Testing

```bash
# full suite
bash scripts/run_all_tests.sh

# individual stages
./build/test_tokenizer    --model models/qwen3-tts-0.6b-f16.gguf
./build/test_encoder      --tokenizer models/qwen3-tts-0.6b-f16.gguf \
                          --audio clone.wav --reference reference/ref_audio_embedding.bin
./build/test_transformer  --model models/qwen3-tts-0.6b-f16.gguf --ref-dir reference/
./build/test_decoder      --tokenizer models/qwen3-tts-tokenizer-f16.gguf \
                          --codes reference/speech_codes.bin --reference reference/decoded_audio.bin

# end-to-end Python vs C++
uv run python scripts/compare_e2e.py
```

Reference results on F16:

- Prefill logits cosine similarity vs Python: `0.99999994`
- Codebook 0 frame-level match: ~81%
- Codebooks 1–4 match: ~84%
- Audio is perceptually equivalent; low waveform correlation is expected due
  to autoregressive divergence under F16.

## Profiling

```bash
cmake -S . -B build -DQWEN3_TTS_TIMING=ON
cmake --build build -j$(nproc)
```

Example output (92 frames, 7.3 s audio):

```
=== Detailed Generation Timing (92 frames) ===

  Prefill:
      Compute:           175.9 ms

  Talker forward_step:
      Compute:          7717.4 ms   (83.9 ms/frame)

  Code predictor:
      Steps (14):      19531.7 ms   (212.3 ms/frame)
      Compute:         20702.6 ms   (225.0 ms/frame)

  Total generate:      28915.0 ms   (3.2 frames/s)
```

The code predictor (15 sequential forward passes per frame) dominates
generation time at ~71%.

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba's Qwen team
- [GGML](https://github.com/ggml-org/ggml) by Georgi Gerganov and contributors
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) vocoder architecture
- Upstream chain:
  [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp)
  → [`khimaros/qwen3-tts.cpp`](https://github.com/khimaros/qwen3-tts.cpp)
  (1.7B support, ICL voice cloning, HTTP server, streaming vocoder, HuggingFace integration)
