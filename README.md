# qwen3-tts.cpp

> ⚠️ **Heads-up:** the changes in this fork were mostly written by Claude
> Code (Anthropic's AI coding agent). It runs and the upstream tests pass,
> but I can't promise production-grade robustness — read the diff and use at
> your own risk.

Text to speech in C++17 on [GGML](https://github.com/ggml-org/ggml), no Python
at inference time. An OpenAI-compatible HTTP server with live streaming and a
voice library, plus a one-shot CLI.

- **Model:** [Qwen3-TTS](https://huggingface.co/collections/khimaros/qwen3-tts), 0.6B or 1.7B
- **Languages:** en, ru, zh, ja, ko, de, fr, es, it, pt
- **Backends:** CUDA, ROCm, Vulkan, Metal, CPU
- **Voice cloning** from a few seconds of reference audio (needs a **Base** model)

## Run it

Three steps: build, start, ask it to speak. Models download themselves on the
first launch (into `~/.cache/huggingface/`).

```bash
git clone https://github.com/Olegd5g56/qwen3-tts.cpp.git
cd qwen3-tts.cpp && git submodule update --init --recursive

cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

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

That is the whole happy path. Everything below is the variations.

### Build for your card

Swap the one flag. On NVIDIA prefer CUDA — it is faster than Vulkan on the same
card and it overlaps the vocoder with generation. **On an AMD RDNA2 card try
both**: Vulkan measured 13% faster than ROCm on an RX 6800 XT, because ROCm's
overlap no longer hides much. See [docs/optimization.md](docs/optimization.md).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release                  # CPU
cmake -S . -B build -DGGML_CUDA=ON   -DCMAKE_BUILD_TYPE=Release # NVIDIA
cmake -S . -B build -DGGML_HIP=ON    -DCMAKE_BUILD_TYPE=Release # AMD ROCm
cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release # Vulkan
```

You get `build/qwen3-tts-server` and `build/qwen3-tts-cli`.

Dependencies (Arch names): `mpg123`, `libopusenc`, `lame`, a C++17 compiler,
CMake 3.14+. Vulkan additionally needs the Vulkan SDK.

Two cards that need an extra flag:

- **CUDA** — `-DCMAKE_CUDA_ARCHITECTURES=` your architecture (75 Turing,
  86 Ampere, 89 Ada, 120 Blackwell). CUDA 13 also needs GCC ≤ 15:
  `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`.
- **ROCm on gfx1030** — `-DAMDGPU_TARGETS=gfx1030
  -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_PREFIX_PATH=/opt/rocm`, and point
  the compilers at `/opt/rocm/llvm/bin/clang{,++}`.

Build options are in [docs/build.md](docs/build.md).

### Or run the container

```bash
docker build -f Dockerfile.cuda   -t qwen3-tts:cuda   .   # NVIDIA
docker build -f Dockerfile.vulkan -t qwen3-tts:vulkan .   # AMD / Intel
docker build -f Dockerfile.cpu    -t qwen3-tts:cpu    .   # CPU

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

Details — build args, the Vulkan image's own GPU access flags, timezones —
in [docs/build.md](docs/build.md).

### Speak in your own voice

Point the server at a directory of voices and name one in the request:

```
voices/
  alice/
    sample.mp3     # 5-15 s of clean speech (or sample.wav)
    sample.txt     # what is said in it — this is what makes cloning work
```

```bash
./build/qwen3-tts-server -m ./models --voices-dir ./voices ...

curl -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "Now in Alice'\''s voice.", "voice": "alice", "language": "en"}' \
  --output alice.wav
```

`sample.txt` must be the exact transcript of `sample.mp3`. Without it the
likeness is much weaker; wrong, and the output is noise. Cloning needs a
**Base** model.

Voice uploads over HTTP, the CLI, and the rest of the library rules are in
[docs/server.md](docs/server.md).

### One-shot, no server

```bash
./build/qwen3-tts-cli -m ./models -t "Hello!" -o hello.wav
```

The output format follows the extension (`.wav`, `.mp3`, `.opus`). Full CLI
reference: [docs/server.md](docs/server.md#cli).

## Which model to download

| Variant | Cloning | `instructions` | Built-in speakers |
|---------|---------|----------------|-------------------|
| 0.6B / 1.7B **Base** | yes | no | no |
| 0.6B / 1.7B **CustomVoice** | no | no | yes |
| 1.7B **VoiceDesign** | no | yes | no |

Take **Base** unless you know you want the others — it is the one that clones
voices, and the voice library is disabled at startup without it.

**1.7B** sounds better; **0.6B** is ~11% faster per frame and needs ~1 GB less
memory. Peak VRAM is ~3.2 GB for the 1.7B and ~2.1 GB for the 0.6B, everything
resident. **Q8_0** is the weight type to take; the reasoning, and what the
4-bit types cost, are in [docs/quantisation.md](docs/quantisation.md).

Ready-made GGUFs: [`khimaros/qwen3-tts`](https://huggingface.co/collections/khimaros/qwen3-tts).
Converting a checkpoint yourself: [docs/models.md](docs/models.md).

## When something is wrong

- The server answers `429` — one request is synthesized at a time; see
  [docs/server.md](docs/server.md).
- Speech ends early, or the tail turns to noise — a known model-level runaway;
  [docs/known-issues.md](docs/known-issues.md) #11.
- It generates noise from the first frame — an F16 talker. Use Q8_0;
  [docs/known-issues.md](docs/known-issues.md) #16.
- Cloning sounds nothing like the reference — `sample.txt` is missing or does
  not match the audio; [docs/known-issues.md](docs/known-issues.md) #23.

## Documentation

| Document | What is in it |
|---|---|
| [docs/server.md](docs/server.md) | Operating manual: every flag, endpoint, request field, the voice library, the CLI, and the runtime environment knobs |
| [docs/build.md](docs/build.md) | Build options, Docker images, testing |
| [docs/models.md](docs/models.md) | Model variants, converting and quantising a checkpoint yourself |
| [docs/architecture.md](docs/architecture.md) | How the pipeline fits together — read this first if you intend to change code |
| [docs/quantisation.md](docs/quantisation.md) | Which weight type to use, decided by listening |
| [docs/optimization.md](docs/optimization.md) | Performance: what was measured, what worked, what was ruled out |
| [docs/streaming_design.md](docs/streaming_design.md) | How the streaming vocoder and live-stream path work |
| [docs/known-issues.md](docs/known-issues.md) | Running log of bugs and rough edges, open and closed |
| [docs/ggml-notes.md](docs/ggml-notes.md) | Findings in ggml itself |
| [docs/tensor_mapping.md](docs/tensor_mapping.md) | HF checkpoint tensor names → GGUF names |
| [AGENTS.md](AGENTS.md) | Conventions for working on the code |

Speed is tracked in `benchmarks/speed.tsv`, one committed row per measured
configuration; `scripts/bench_speed.sh` writes them. The protocol, and how to
compare two rows honestly, is in
[docs/optimization.md](docs/optimization.md#tracking-speed-across-commits).

## Acknowledgments

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) — Alibaba Qwen team
- [GGML](https://github.com/ggml-org/ggml) — Georgi Gerganov & contributors
- [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) — vocoder architecture
- Upstream chain:
  [`predict-woo/qwen3-tts.cpp`](https://github.com/predict-woo/qwen3-tts.cpp)
  → [`khimaros/qwen3-tts.cpp`](https://github.com/khimaros/qwen3-tts.cpp)
  (1.7B support, ICL voice cloning, HTTP server, streaming vocoder, HF integration)
