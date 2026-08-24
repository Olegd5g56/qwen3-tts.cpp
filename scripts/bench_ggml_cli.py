#!/usr/bin/env python3
"""
Per-stage speed benchmark of this fork's CLI, in the shape `bench_torch_vs_ggml.py`
reports, so the two can be put in one table.

Runs the CLI with the stage pipeline off (`QWEN3_TTS_PIPELINE=0`) so generate and
decode time apart instead of overlapping, then parses the verbose log for the
stage timings and the frame counts.

Usage:
    python3 scripts/bench_ggml_cli.py \
        --model /path/to/Qwen3-TTS-12Hz-0.6B-Base-Q8_0.gguf \
        --vocoder /path/to/Qwen3-TTS-Tokenizer-12Hz-F16.gguf \
        --text /storage/neocortex/AI/tts_test/ward.txt \
        --ref-audio examples/readme_clone_input.wav \
        --ref-text reference_text.txt \
        --language ru --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RE_TIMING = re.compile(
    r"timing:\s*tokenize=(\d+)\s*ms\s+encode=(\d+)\s*ms\s+generate=(\d+)\s*ms\s+"
    r"decode=(\d+)\s*ms\s+total=(\d+)\s*ms"
)
RE_ICL = re.compile(r"\[icl\]\s*ref_frames=(\d+)\s+new_frames=(\d+)\s+prepended=(\d+)")
RE_CODES = re.compile(r"speech codes generated:\s*(\d+)\s*frames")
RE_AUDIO = re.compile(r"audio:\s*([\d.]+)s duration")
# Only present when the CLI was built with QWEN3_TTS_TIMING.
RE_TALKER = re.compile(r"talker forward:\s*total=[\d.]+ ms \(([\d.]+)/frame\)")
RE_CODEPRED = re.compile(r"code predictor:\s*total=[\d.]+ ms \(([\d.]+)/frame\)")


def _read_arg_text(value: str) -> str:
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value


def parse_log(log: str) -> dict[str, Any] | None:
    m = RE_TIMING.search(log)
    if not m:
        return None
    tokenize, encode, generate, decode, total = (int(g) for g in m.groups())

    icl = RE_ICL.search(log)
    codes = RE_CODES.search(log)
    if icl:
        n_ref, n_new = int(icl.group(1)), int(icl.group(2))
    else:
        n_ref, n_new = 0, int(codes.group(1)) if codes else 0
    n_dec = n_ref + n_new

    audio = RE_AUDIO.search(log)
    talker = RE_TALKER.search(log)
    codepred = RE_CODEPRED.search(log)

    extra: dict[str, Any] = {}
    if talker:
        extra["talker_ms_per_frame"] = float(talker.group(1))
    if codepred:
        extra["code_predictor_ms_per_frame"] = float(codepred.group(1))

    return {
        **extra,
        "tokenize_ms": tokenize,
        "prompt_ms": encode,
        "generate_ms": generate,
        "decode_ms": decode,
        "total_ms": total,
        "n_ref_frames": n_ref,
        "n_new_frames": n_new,
        "n_decoded_frames": n_dec,
        "generate_ms_per_frame": generate / max(n_new, 1),
        "decode_ms_per_decoded_frame": decode / max(n_dec, 1),
        "decode_ms_per_new_frame": decode / max(n_new, 1),
        "audio_seconds": float(audio.group(1)) if audio else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", default=str(PROJECT_ROOT / "build-cuda" / "qwen3-tts-cli"), type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--text", required=True, help="text to synthesize, or a path to a file holding it")
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text", default="", help="reference transcript, or a path to a file holding it")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-reseeds", type=int, default=4, help="extra attempts to spend on run-aways")
    ap.add_argument("--pipeline", action="store_true", help="leave the generate/decode overlap on")
    ap.add_argument("--wav-out", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    text = _read_arg_text(args.text)
    ref_text = _read_arg_text(args.ref_text) if args.ref_text else ""

    # The CLI picks the output format from the extension, so a throwaway still
    # has to end in .wav.
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    wav_out = args.wav_out or Path(tmp_wav.name)

    base_cmd = [
        str(args.cli),
        "-m", args.model,
        "--vocoder", args.vocoder,
        "-t", text,
        "-r", args.ref_audio,
        "-l", args.language,
        "-j", str(args.threads),
        "-o", str(wav_out),
    ]
    if ref_text:
        base_cmd += ["--ref-text", ref_text]

    env_extra = {"QWEN3_TTS_PIPELINE": "1" if args.pipeline else "0", "TTS_VERBOSE": "1"}

    runs: list[dict[str, Any]] = []
    seed = args.seed
    attempts_left = args.runs + 1 + args.max_reseeds
    # One warm-up so the first run's cold caches and lazy loads stay out of the numbers.
    i = 0
    while i < args.runs + 1 and attempts_left > 0:
        attempts_left -= 1
        cmd = base_cmd + ["--seed", str(seed)]
        seed += 1
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            env={**dict(__import__("os").environ), **env_extra},
        )
        log = proc.stdout + proc.stderr
        # A run-away (no end-of-speech before the frame budget) is a sampling
        # outcome, not a timing sample: it times a different regime and the CLI
        # exits non-zero. Re-roll the seed instead of dying.
        if "ran away" in log or "did not signal end of speech" in log:
            print(f"[reseed] run-away at seed {seed - 1}, retrying", flush=True)
            continue
        if proc.returncode != 0:
            print(log[-3000:], file=sys.stderr)
            print(f"CLI failed with code {proc.returncode}", file=sys.stderr)
            return 1
        r = parse_log(log)
        if r is None:
            print(log[-3000:], file=sys.stderr)
            print("could not parse the timing line from the CLI log", file=sys.stderr)
            return 1
        i += 1
        if i == 1:
            print("[warmup] done", flush=True)
            continue
        runs.append(r)
        print(
            f"[run {len(runs)}/{args.runs}] frames={r['n_new_frames']:4d} "
            f"generate={r['generate_ms_per_frame']:6.2f} ms/frame  "
            f"decode={r['decode_ms_per_decoded_frame']:6.2f} ms/decoded-frame  "
            f"prompt={r['prompt_ms']:6.1f} ms  audio={r['audio_seconds']:.2f}s",
            flush=True,
        )

    if len(runs) < args.runs:
        print(f"only {len(runs)}/{args.runs} runs produced a timing sample", file=sys.stderr)
        if not runs:
            return 1

    def med(key: str) -> float:
        return float(np.median([r[key] for r in runs]))

    summary = {k: med(k) for k in runs[0] if k != "sample_rate"}
    print("\n=== median over %d runs ===" % args.runs)
    for k, v in summary.items():
        print(f"{k:32s} {v:10.2f}")

    if args.json:
        args.json.write_text(
            json.dumps({"env": {"cli": str(args.cli), "model": args.model, "pipeline": args.pipeline},
                        "runs": runs, "summary": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
