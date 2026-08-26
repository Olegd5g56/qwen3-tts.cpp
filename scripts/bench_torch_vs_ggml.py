#!/usr/bin/env python3
"""
Per-stage speed benchmark of the reference PyTorch pipeline (`qwen-tts`).

Times the same three stages this fork's CLI reports, so the two can be put in
one table:

    prompt   -- reference audio -> x-vector + ICL codes  (C++: "encode")
    generate -- talker + code predictor -> speech codes  (C++: "generate")
    decode   -- speech codes -> waveform                 (C++: "decode")

Frame counts differ between the two engines because sampling does, so the
comparable number is ms/frame, never wall time. `decode` covers the ICL
reference frames as well as the generated ones -- the C++ side does the same,
so the per-frame figures line up.

Usage:
    .venv/bin/python scripts/bench_torch_vs_ggml.py \
        --model models/Qwen3-TTS-12Hz-0.6B-Base \
        --text /storage/neocortex/AI/tts_test/ward.txt \
        --ref-audio benchmarks/voice/sample.mp3 \
        --ref-text "$(cat benchmarks/voice/sample.txt)" \
        --language russian --dtype float16 --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

# Short and unrelated to the benchmark text: just enough to force every kernel
# to be compiled and every lazy buffer to be allocated before we time anything.
WARMUP_TEXT = "Warm up."


def _read_arg_text(value: str) -> str:
    """Accept either a literal string or a path to a file holding one."""
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value


class Stopwatch:
    """Wall time around a block, with the CUDA queue drained at both ends."""

    def __init__(self, cuda: bool) -> None:
        self.cuda = cuda
        self.ms = 0.0

    def __enter__(self) -> "Stopwatch":
        if self.cuda:
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        if self.cuda:
            torch.cuda.synchronize()
        self.ms = (time.perf_counter() - self._t0) * 1000.0


def run_once(model: Any, args: argparse.Namespace, text: str, ref_text: str, cuda: bool) -> dict[str, Any]:
    """One full synthesis, timed stage by stage.

    Mirrors `Qwen3TTSModel.generate_voice_clone` step for step; it is inlined
    here only so each stage can be timed on its own.
    """
    with Stopwatch(cuda) as sw_prompt:
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=str(args.ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=args.x_vector_only,
        )
        prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)

    input_ids = model._tokenize_texts([model._build_assistant_text(text)])
    ref_ids = [model._tokenize_texts([model._build_ref_text(ref_text)])[0]] if ref_text else [None]
    gen_kwargs = model._merge_generate_kwargs()

    with Stopwatch(cuda) as sw_gen:
        codes_list, _ = model.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=prompt,
            languages=[args.language],
            non_streaming_mode=args.non_streaming,
            **gen_kwargs,
        )

    codes = codes_list[0]
    ref_codes = (prompt.get("ref_code") or [None])[0]
    if ref_codes is not None:
        codes_for_decode = torch.cat([ref_codes.to(codes.device), codes], dim=0)
    else:
        codes_for_decode = codes

    with Stopwatch(cuda) as sw_dec:
        wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])

    wav = wavs[0]
    n_ref = int(ref_codes.shape[0]) if ref_codes is not None else 0
    n_new = int(codes.shape[0])
    n_dec = int(codes_for_decode.shape[0])

    # Trim the reference prefix back off, exactly as generate_voice_clone does,
    # so the reported duration is the audio a caller would actually get.
    cut = int(n_ref / max(n_dec, 1) * wav.shape[0]) if n_ref else 0
    out = wav[cut:]

    return {
        "prompt_ms": sw_prompt.ms,
        "generate_ms": sw_gen.ms,
        "decode_ms": sw_dec.ms,
        "total_ms": sw_prompt.ms + sw_gen.ms + sw_dec.ms,
        "n_ref_frames": n_ref,
        "n_new_frames": n_new,
        "n_decoded_frames": n_dec,
        "generate_ms_per_frame": sw_gen.ms / max(n_new, 1),
        "decode_ms_per_decoded_frame": sw_dec.ms / max(n_dec, 1),
        "decode_ms_per_new_frame": sw_dec.ms / max(n_new, 1),
        "audio_seconds": len(out) / sr,
        "sample_rate": sr,
        "_wav": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(PROJECT_ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"))
    ap.add_argument("--text", required=True, help="text to synthesize, or a path to a file holding it")
    ap.add_argument("--ref-audio", required=True, type=Path)
    ap.add_argument("--ref-text", default="", help="reference transcript, or a path to a file holding it")
    ap.add_argument("--language", default="russian")
    ap.add_argument("--dtype", default="float16", choices=sorted(DTYPES))
    ap.add_argument("--attn", default="sdpa", help="attn_implementation (flash_attention_2 needs Ampere+)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--runs", type=int, default=3, help="timed runs after the warm-up")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--x-vector-only", action="store_true", help="skip ICL; clone from the x-vector alone")
    ap.add_argument(
        "--non-streaming",
        action="store_true",
        help="put the whole text in the prefill. The C++ side feeds text one token per frame, "
        "so leave this off for a like-for-like comparison; with it on the model stops early.",
    )
    ap.add_argument("--wav-out", type=Path, help="write the last run's audio here")
    ap.add_argument("--json", type=Path, help="write the full result set here")
    args = ap.parse_args()

    text = _read_arg_text(args.text)
    ref_text = _read_arg_text(args.ref_text) if args.ref_text else ""

    cuda = args.device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        print("CUDA requested but not available", file=sys.stderr)
        return 1

    from qwen_tts import Qwen3TTSModel

    if cuda:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=DTYPES[args.dtype],
        attn_implementation=args.attn,
    )
    if cuda:
        torch.cuda.synchronize()
    load_ms = (time.perf_counter() - t0) * 1000.0

    env = {
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": args.dtype,
        "attn": args.attn,
        "model": str(args.model),
        "language": args.language,
        "icl": not args.x_vector_only,
        "text_chars": len(text),
        "load_ms": load_ms,
    }
    print(json.dumps(env, ensure_ascii=False, indent=2))

    torch.manual_seed(args.seed)
    print(f"\n[warmup] ...", flush=True)
    run_once(model, args, WARMUP_TEXT, ref_text, cuda)

    if cuda:
        torch.cuda.reset_peak_memory_stats()

    runs: list[dict[str, Any]] = []
    for i in range(args.runs):
        torch.manual_seed(args.seed + i)
        r = run_once(model, args, text, ref_text, cuda)
        wav = r.pop("_wav")
        runs.append(r)
        print(
            f"[run {i + 1}/{args.runs}] frames={r['n_new_frames']:4d} "
            f"generate={r['generate_ms_per_frame']:6.2f} ms/frame  "
            f"decode={r['decode_ms_per_decoded_frame']:6.2f} ms/decoded-frame  "
            f"prompt={r['prompt_ms']:6.1f} ms  audio={r['audio_seconds']:.2f}s",
            flush=True,
        )

    if args.wav_out:
        sf.write(str(args.wav_out), wav, runs[-1]["sample_rate"])

    def med(key: str) -> float:
        return float(np.median([r[key] for r in runs]))

    summary = {
        "generate_ms_per_frame": med("generate_ms_per_frame"),
        "decode_ms_per_decoded_frame": med("decode_ms_per_decoded_frame"),
        "decode_ms_per_new_frame": med("decode_ms_per_new_frame"),
        "prompt_ms": med("prompt_ms"),
        "total_ms": med("total_ms"),
        "n_new_frames_median": med("n_new_frames"),
        "audio_seconds_median": med("audio_seconds"),
        "peak_vram_mib": (torch.cuda.max_memory_allocated() / 2**20) if cuda else 0.0,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    print("\n=== median over %d runs ===" % args.runs)
    for k, v in summary.items():
        print(f"{k:32s} {v:10.2f}")

    if args.json:
        args.json.write_text(
            json.dumps({"env": env, "runs": runs, "summary": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
