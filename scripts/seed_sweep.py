#!/usr/bin/env python3
"""
Seeded transcription test: does this weight type still say the words?

Why this exists. `qwen3-tts-quantize --verify` ranks weight types by relative
rms error of the weights. That is a ranking of *distortion*, not of behaviour,
and on 2026-08-26 it ranked two types backwards: q4_k has less weight error
than q4_0 at the same 4.5 bits, and q4_k was the one that dropped a whole word
from a sentence 15 times in 20 while q4_0 never did once. Listening cannot
catch that either, because every weight type draws a different token sequence
from the same seed, so an A/B is one lottery ticket against another.

What does catch it is a rate. Give every candidate the same line with seeds
1..N, transcribe all of it with the same judge, and score whether the words
came back. One draw is a lottery; sixty draws is a rate with a confidence
interval, and two rates can be compared.

Sixty, not twenty, when the question is "is there any difference at all":
near the ceiling twenty draws cannot separate 98% from 92%.

The judge is the stack's whisper (`cd $AI/AI && docker compose up -d
stt-server`), which publishes no host port - POST to its container IP on 8082.
It is a noisy judge, but it is the same judge for every row, and the comparison
is between rows.

Screen the line on bf16 first. A construct the model fails at full precision
ranks nothing: the 0.6B returns "17,5" only 7 times in 60 at bf16, so that line
cannot rank anything on the 0.6B.

**Calibrate the pattern against the judge before reading the numbers.** whisper
writes one spoken number several ways: across 180 clips of the same sentence
"семнадцать целых пять" came back as `17,5` 97 times, `17-5` 9, `17, 5` 4,
`17. 5` 2, plus a `17,55`. Scoring only the literal `17,5` charged all of those
to the model and put bf16 at 47/60 when it is really 54/60. Dump the distinct
renderings first (`cut -f4 transcripts.tsv | sort | uniq -c`), then write the
pattern. Use `--expect-re` when the alternatives are a family rather than a
list.

Usage:

    scripts/seed_sweep.py \\
        --cli build-hip-native/qwen3-tts-cli \\
        --vocoder $D/Qwen3-TTS-Tokenizer-12Hz-F16.gguf \\
        --voice ostro --voices-dir /storage/neocortex/AI/Templates/voices \\
        --lang ru --seeds 60 \\
        --line 'В 2026 году цена выросла на 17,5 процента, до 3480 рублей.' \\
        --expect '17,5|17.5' --expect 2026 --expect 3480 \\
        --model bf16=$D/Qwen3-TTS-12Hz-1.7B-Base-BF16.gguf \\
        --model q4_k=$D/Qwen3-TTS-12Hz-1.7B-Base-Q4_K.gguf \\
        --model q4_0=$D/Qwen3-TTS-12Hz-1.7B-Base-Q4_0.gguf \\
        --out /tmp/sweep --stt http://172.21.0.2:8082

Writes <out>/transcripts.tsv (model, seed, expectation hits, transcript) and
prints the scored table. Re-running reuses clips and transcripts already on
disk, so an interrupted sweep resumes.
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path


def wilson(hits: int, n: int) -> tuple[float, float]:
    """95% Wilson interval. Normal approximation puts 20/20 at 100+-0%."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959964
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - s) / d), min(1.0, (c + s) / d))


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a,b],[c,d]]. Small tables, exact sum."""
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x: int) -> float:
        return (
            math.comb(row1, x)
            * math.comb(n - row1, col1 - x)
            / math.comb(n, col1)
        )

    lo = max(0, col1 - (n - row1))
    hi = min(row1, col1)
    observed = prob(a)
    # Guard against float noise excluding the observed table itself.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= observed * (1 + 1e-9)))


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def scored(transcript: str, expectations: list[tuple[str, bool]]) -> list[bool]:
    t = normalise(transcript)
    return [bool(re.search(e, t)) if is_re
            else any(alt.lower() in t for alt in e.split("|"))
            for e, is_re in expectations]


def synth(cli: Path, model: Path, vocoder: Path, voice: str, voices_dir: str,
          lang: str, line: str, seed: int, out: Path) -> bool:
    cmd = [str(cli), "-m", str(model), "--vocoder", str(vocoder),
           "-v", voice, "-l", lang, "--seed", str(seed),
           "-t", line, "-o", str(out)]
    if voices_dir:
        cmd += ["--voices-dir", voices_dir]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # A runaway exits non-zero and writes nothing; that is a result, not a
    # crash, and it must not be silently scored as a miss.
    return r.returncode == 0 and out.exists()


def transcribe(stt: str, wav: Path, lang: str) -> str:
    r = subprocess.run(
        ["curl", "-s", "-m", "120", "-X", "POST",
         f"{stt.rstrip('/')}/v1/audio/transcriptions",
         "-F", f"file=@{wav}", "-F", "response_format=text",
         "-F", f"language={lang}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"stt request failed for {wav}: {r.stderr}")
    return normalise(r.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--vocoder", required=True)
    ap.add_argument("--model", action="append", required=True,
                    metavar="LABEL=PATH")
    ap.add_argument("--line", required=True)
    ap.add_argument("--expect", action="append", default=[],
                    metavar="ALT|ALT", help="substring, alternatives on |")
    ap.add_argument("--expect-re", action="append", default=[],
                    metavar="REGEX", help="python regex, matched on the "
                                          "lowercased whitespace-collapsed text")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--seed-start", type=int, default=1,
                    help="first seed; use a fresh range to replicate a result "
                         "on seeds the first run never saw")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--voices-dir", default="")
    ap.add_argument("--lang", default="ru")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stt", required=True)
    args = ap.parse_args()
    expectations = ([(e, False) for e in args.expect]
                    + [(e, True) for e in args.expect_re])
    if not expectations:
        raise SystemExit("need at least one --expect or --expect-re")
    names = [e for e, _ in expectations]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [(m.split("=", 1)[0], Path(m.split("=", 1)[1])) for m in args.model]
    for _, path in models:
        if not path.exists():
            raise SystemExit(f"no such model: {path}")

    tsv = out / "transcripts.tsv"
    done: dict[tuple[str, int], str] = {}
    if tsv.exists():
        for row in tsv.read_text().splitlines()[1:]:
            f = row.split("\t")
            if len(f) >= 3:
                done[(f[0], int(f[1]))] = f[-1]

    runaways: dict[str, int] = {label: 0 for label, _ in models}
    lines = ["\t".join(["model", "seed"]
                       + [f"has:{e}" for e in names] + ["transcript"])]

    for label, path in models:
        for seed in range(args.seed_start,
                          args.seed_start + args.seeds):
            key = (label, seed)
            if key in done:
                text = done[key]
            else:
                wav = out / f"{label}-{seed}.wav"
                if not wav.exists() and not synth(
                        Path(args.cli), path, Path(args.vocoder), args.voice,
                        args.voices_dir, args.lang, args.line, seed, wav):
                    runaways[label] += 1
                    print(f"  {label} seed {seed}: no audio (runaway?)",
                          file=sys.stderr)
                    continue
                text = transcribe(args.stt, wav, args.lang)
            hits = scored(text, expectations)
            lines.append("\t".join([label, str(seed)]
                                   + ["1" if h else "0" for h in hits]
                                   + [text]))
            print(f"  {label} seed {seed}"
                  f" {''.join('+' if h else '.' for h in hits)}",
                  file=sys.stderr)
    tsv.write_text("\n".join(lines) + "\n")

    print(f"\nline: {args.line}")
    print(f"seeds: {args.seed_start}..{args.seed_start + args.seeds - 1}"
          f"   voice: {args.voice}   judge: whisper\n")

    counts: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    for row in lines[1:]:
        f = row.split("\t")
        label = f[0]
        counts.setdefault(label, [0] * len(names))
        totals[label] = totals.get(label, 0) + 1
        for i in range(len(names)):
            counts[label][i] += int(f[2 + i])

    head = f"{'weights':<10}{'n':>4}  " + "  ".join(
        f"{e[:14]:>14}" for e in names)
    print(head)
    print("-" * len(head))
    for label, _ in models:
        if label not in counts:
            continue
        n = totals[label]
        cells = []
        for i in range(len(names)):
            h = counts[label][i]
            lo, hi = wilson(h, n)
            cells.append(f"{h:>3}/{n:<3} {lo*100:>3.0f}-{hi*100:<3.0f}%")
        extra = f"  ({runaways[label]} runaway)" if runaways[label] else ""
        print(f"{label:<10}{n:>4}  " + "  ".join(cells) + extra)

    if len(models) > 1:
        print("\nFisher exact, two-sided, against the first row:")
        base = models[0][0]
        for label, _ in models[1:]:
            if label not in counts:
                continue
            for i, e in enumerate(names):
                a, n1 = counts[base][i], totals[base]
                c, n2 = counts[label][i], totals[label]
                p = fisher(a, n1 - a, c, n2 - c)
                flag = "  <-- significant" if p < 0.05 else ""
                print(f"  {e[:14]:>14}  {base} {a}/{n1} vs {label} {c}/{n2}"
                      f"   p={p:.3f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
