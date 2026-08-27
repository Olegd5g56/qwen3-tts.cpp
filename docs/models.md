# Models — variants, files, and rolling your own

Which checkpoint does what, where the GGUFs come from, and how to convert or
re-quantise one yourself. Which *weight type* to pick is a separate question
with a separate answer: [quantisation.md](quantisation.md).

## Variants

| Variant | Cloning | `instructions` | Built-in speakers |
|---------|---------|----------------|-------------------|
| 0.6B / 1.7B **Base** | yes | no | no |
| 0.6B / 1.7B **CustomVoice** | no | no | yes |
| 1.7B **VoiceDesign** | no | yes | no |
| **vocoder** (Qwen3-TTS-Tokenizer-12Hz) | — | — | — |

Voice cloning and the voice library need a **Base** model; loading anything
else disables the library at startup and says so.

Every setup needs two files: a talker and the vocoder. The vocoder is shared
across variants and sizes.

## 0.6B or 1.7B

The 1.7B sounds better. The 0.6B is worth **11–13% per frame**, not the ~50%
its parameter count suggests, because two of the three stages — the code
predictor and the vocoder — are the same size in both. Treat it as a memory
choice, not a speed one.

| Model | Peak VRAM, long clip |
|---|---|
| 1.7B | ~3.2 GB |
| 0.6B | ~2.1 GB |

That is with everything resident. Budget it plus whatever else is on the card;
a CUDA out-of-memory aborts the process rather than degrading. To cut it:
`QWEN3_TTS_DECODE_BATCH=8` (−120 MB, −7% throughput) or `QWEN3_TTS_LOW_MEM=1`
(much lower peak, one model load per request). Both are in
[server.md](server.md#environment-knobs).

## Getting the files

**Ready-made:** [`khimaros/qwen3-tts`](https://huggingface.co/collections/khimaros/qwen3-tts),
F16 and Q8_0.

**Auto-download:** `--hf-repo <repo>[:<quant>]` and `--hf-repo-v <repo>[:<quant>]`
on either front end. The default quant is `Q8_0` and files are cached in
`~/.cache/huggingface/`.

**Local:** `-m talker.gguf -v vocoder.gguf`. `-m` also takes a directory and
finds the vocoder inside it.

## Converting a checkpoint

```bash
scripts/convert_tts_to_gguf.py --type q8_0 <hf-checkpoint-dir> talker.gguf
```

`--type` takes `q8_0` (the default), `bf16`, `q4_0` and `f32`.

**Do not use `f16`.** This model's activations do not fit in it and the result
generates noise from the first frame — the converter warns, and
[known-issues.md](known-issues.md) #16 has the numbers. It is kept only so the
failure can be reproduced.

## Re-quantising

The Python converter cannot write K-quants. Convert to `bf16` first, then:

```bash
build/qwen3-tts-quantize talker-bf16.gguf talker-q4_k.gguf q4_k --verify
```

`--verify` prints how much each tensor lost. `--tensor-type SUBSTR=TYPE`
overrides individual tensors and beats the built-in skip list — use it to hold
a tensor at source type, or to force one the list would otherwise skip.

Which type to actually ship is not the usual llama.cpp answer, and it was
decided by listening rather than by perplexity: [quantisation.md](quantisation.md).
