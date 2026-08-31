# Building — options, images, tests

The README has the four build lines that cover most machines. This is
everything else.

## Build options

| Option | Default | Effect |
|---|---|---|
| `QWEN3_TTS_NATIVE` | `OFF` | `-march=native` for this CPU, and pins ggml's `GGML_NATIVE` to the same value. Turn it on only when the build host and the run host are the same machine — elsewhere the binary dies with `SIGILL`. `OFF` still assumes AVX2. |
| `QWEN3_TTS_SERVER` | `ON` | `OFF` skips building the server. |
| `QWEN3_TTS_TIMING` | `OFF` | Compiles in per-stage timing counters. |
| `QWEN3_TTS_COREML` | `ON` on Apple | Builds the Core ML code-predictor bridge. |

`GGML_NATIVE` is not set directly: the project forces it from
`QWEN3_TTS_NATIVE` before adding the ggml subdirectory, because ggml defaults
it to `ON` and would otherwise build a CPU backend for the build machine
whatever this project asked for. A build directory configured before that rule
existed keeps whatever it was configured with — re-running `cmake -S . -B <dir>`
brings it in line.

**Before comparing two build directories, check they agree**:
`grep GGML_NATIVE */CMakeCache.txt`. A day of backend comparison was once spent
measuring that flag disagreeing between two trees, which nothing warns about.
(The flag itself turned out not to matter — see
[optimization.md](optimization.md).)

ggml is vendored as a submodule at `./ggml` and built as part of this project
via `add_subdirectory`. There is no separate ggml build step; backend options
like `GGML_CUDA`, `GGML_HIP`, `GGML_VULKAN`, `GGML_METAL` pass straight through.

## Docker

```bash
docker build -f Dockerfile.cuda   -t qwen3-tts:cuda   .   # NVIDIA
docker build -f Dockerfile.vulkan -t qwen3-tts:vulkan .   # AMD / Intel
docker build -f Dockerfile.cpu    -t qwen3-tts:cpu    .   # CPU
```

Build args: `QWEN3_TTS_NATIVE=ON` (see above — only for an image that will run
on the machine that built it), and for `Dockerfile.cuda` also `CUDA_ARCH=75`
(the default). `Dockerfile.cuda` carries a `HEALTHCHECK`.

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

The Vulkan image needs `--device /dev/dri --group-add video` instead of
`--gpus`. Set `-e TZ=<IANA name>` to match log timestamps to the host.

**An image and a host build of the same commit are two different
configurations.** The image ships its own userspace GPU driver — the Vulkan one
carries Debian's mesa, not the host's — so a driver-sensitive measurement on
one says nothing about the other. `scripts/bench_speed.sh --docker <image>`
benchmarks an image directly for that reason.

## Testing

```bash
ctest --test-dir build

QWEN3_TTS_TEST_MODEL=/path/to/talker.gguf \
QWEN3_TTS_TEST_VOCODER=/path/to/tokenizer.gguf \
  ctest --test-dir build
```

Tests whose fixtures are missing report **Skipped** (exit 77) rather than
failing, so a red run is always a real regression. Some need only a model —
those two variables — and some need reference dumps produced by the original
Python implementation, which are not in the repo;
`scripts/generate_deterministic_reference.py` regenerates them.

| Test | Needs | What it protects |
|---|---|---|
| `tokenizer_test` | nothing | BPE tokenization |
| `encoder_test` | model + reference dump | speaker encoder output |
| `transformer_test` | model + reference dump | deterministic generation |
| `streaming_parity_test` | model | streaming decode == one-shot decode |
| `c_api_test` | model | the C ABI: ICL, streaming, voice caching, struct-size guard |

All five are green on every backend. Anything red is new.

`scripts/run_all_tests.sh` **does not run** — see `known-issues.md`, open rough
edges. It is hard-wired to a `./build/` directory and a `models/` layout the
repo stopped using, so every section skips and the one that does not, fails.
ctest is the whole automated suite today.
