#!/usr/bin/env bash
# Whole-request speed benchmark, appended to a versioned history.
#
# Why this exists: on 2026-08-25 a day of cross-backend comparison turned out to
# be measuring GGML_NATIVE disagreeing between two build directories. Nothing
# warns you about that. So every row this writes carries the commit, the build
# flags, the backend, the model and the text, and a comparison is only honest
# between rows where all of those match except the one being studied.
#
# The protocol, and why each part of it is there:
#
#   * A warm server, not the CLI. The CLI pays model load, kernel/shader
#     compilation and the ICL warm-up on every invocation; the deployment pays
#     them once. The CLI also defaults to one-shot decode where the server
#     streams.
#   * One warm-up request, discarded. It costs about double (2026-08-25: 79.9
#     vs ~32 ms/frame on CUDA).
#   * Four timed requests, averaged, all of them recorded.
#   * Whole request wall time. Do NOT compare backends by generation ms/frame:
#     the vocoder is overlapped into generation on CUDA/ROCm/Metal and not on
#     Vulkan/CPU, so that number means different things on either side.
#   * One voice in the voices directory. The server preloads the whole library
#     in the background and 198 voices compete with what is being measured.
#
# Usage:
#   scripts/bench_speed.sh --build build-cuda --label "CUDA 1660S" [options]
#
#   --build DIR       build directory holding qwen3-tts-server   (required)
#   --label NAME      how this configuration is named in the log (required)
#   --env K=V         extra environment for the server (repeatable),
#                     e.g. --env CUDA_VISIBLE_DEVICES=0
#   --model PATH      talker gguf      (default: $TTS_MODEL)
#   --vocoder PATH    vocoder gguf     (default: $TTS_VOCODER)
#   --voice NAME      voice to use     (default: $TTS_BENCH_VOICE, else the
#                     first in --voices-dir; it is recorded in the row, because
#                     a different voice speaks at a different rate and so
#                     produces a different number of frames for the same text)
#   --voices-dir DIR  voice library    (default: $TTS_VOICES_DIR)
#   --text FILE       input text       (default: benchmarks/bench_ru.txt)
#   --runs N          timed runs after the warm-up (default: 4)
#   --port N          (default: 8099)
#   --out FILE        history file     (default: benchmarks/speed.tsv)
#   --note TEXT       free-form note stored with the row
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD= LABEL= MODEL=${TTS_MODEL:-} VOCODER=${TTS_VOCODER:-} VOICE=${TTS_BENCH_VOICE:-} VOICES_DIR=${TTS_VOICES_DIR:-}
TEXT="$ROOT/benchmarks/bench_ru.txt" RUNS=4 PORT=8099 OUT="$ROOT/benchmarks/speed.tsv" NOTE=
declare -a EXTRA_ENV=()

while [ $# -gt 0 ]; do
    case "$1" in
        --build)       BUILD=$2; shift 2 ;;
        --label)       LABEL=$2; shift 2 ;;
        --env)         EXTRA_ENV+=("$2"); shift 2 ;;
        --model)       MODEL=$2; shift 2 ;;
        --vocoder)     VOCODER=$2; shift 2 ;;
        --voice)       VOICE=$2; shift 2 ;;
        --voices-dir)  VOICES_DIR=$2; shift 2 ;;
        --text)        TEXT=$2; shift 2 ;;
        --runs)        RUNS=$2; shift 2 ;;
        --port)        PORT=$2; shift 2 ;;
        --out)         OUT=$2; shift 2 ;;
        --note)        NOTE=$2; shift 2 ;;
        -h|--help)     sed -n '2,50p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "bench_speed: $*" >&2; exit 1; }
[ -n "$BUILD" ]  || die "--build is required"
[ -n "$LABEL" ]  || die "--label is required"
[ -x "$BUILD/qwen3-tts-server" ] || die "no qwen3-tts-server in $BUILD"
[ -n "$MODEL" ] && [ -f "$MODEL" ] || die "--model / TTS_MODEL must point at a gguf"
[ -n "$VOCODER" ] && [ -f "$VOCODER" ] || die "--vocoder / TTS_VOCODER must point at a gguf"
[ -f "$TEXT" ] || die "text file not found: $TEXT"
[ -n "$VOICES_DIR" ] && [ -d "$VOICES_DIR" ] || die "--voices-dir / TTS_VOICES_DIR must be a directory"
command -v jq   >/dev/null || die "jq is required"
command -v curl >/dev/null || die "curl is required"

# One voice only: the background library preload competes with the measurement.
[ -n "$VOICE" ] || VOICE=$(ls "$VOICES_DIR" | head -1)
[ -n "$VOICE" ] || die "no voice found in $VOICES_DIR"
ONEVOICE=$(mktemp -d); trap 'rm -rf "$ONEVOICE"' EXIT
ln -s "$VOICES_DIR/$VOICE" "$ONEVOICE/$VOICE" || die "cannot link voice $VOICE"

cache() { grep -E "^$1:" "$BUILD/CMakeCache.txt" 2>/dev/null | cut -d= -f2- ; }
COMMIT=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$ROOT" diff --quiet 2>/dev/null || COMMIT="$COMMIT-dirty"
NATIVE=$(cache GGML_NATIVE); TIMING=$(cache QWEN3_TTS_TIMING)
TEXT_ID="$(basename "$TEXT"):$(wc -c < "$TEXT" | tr -d ' ')B"

"$BUILD/qwen3-tts-server" --help >/dev/null 2>&1  # fail fast on a broken binary
env "${EXTRA_ENV[@]}" \
    TTS_MODEL="$MODEL" TTS_VOCODER="$VOCODER" TTS_VOICES_DIR="$ONEVOICE" \
    TTS_PORT="$PORT" TTS_HOST=127.0.0.1 TTS_VERBOSE=1 TTS_IDLE_TIMEOUT=3600 \
    "$BUILD/qwen3-tts-server" > "$ONEVOICE/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; rm -rf "$ONEVOICE"' EXIT

for _ in $(seq 1 240); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SERVER 2>/dev/null || { cat "$ONEVOICE/server.log" >&2; die "server exited"; }
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "server never became healthy"

request() {
    jq -n --rawfile input "$TEXT" \
        '{input: $input, voice: "'"$VOICE"'", response_format: "wav"}' \
    | curl -s -m 900 -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/audio/speech" \
        -H 'Content-Type: application/json' -d @-
}

printf 'bench_speed: %s  commit=%s voice=%s native=%s timing=%s\n' \
    "$LABEL" "$COMMIT" "$VOICE" "${NATIVE:-?}" "${TIMING:-?}"
request  # warm-up, discarded: it costs about double
times=()
for i in $(seq 1 "$RUNS"); do
    start=$(date +%s.%N); request; end=$(date +%s.%N)
    t=$(LC_ALL=C awk "BEGIN{printf \"%.2f\", $end - $start}")
    times+=("$t"); printf '  run %d: %s s\n' "$i" "$t"
done
mean=$(printf '%s\n' "${times[@]}" | LC_ALL=C awk '{s+=$1}END{printf "%.2f", s/NR}')

[ -s "$OUT" ] || printf 'date\tcommit\tlabel\tmodel\ttext\tvoice\tnative\ttiming\tmean_s\truns_s\tnote\n' > "$OUT"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%MZ)" "$COMMIT" "$LABEL" "$(basename "$MODEL")" "$TEXT_ID" "$VOICE" \
    "${NATIVE:-?}" "${TIMING:-?}" "$mean" "$(IFS=,; echo "${times[*]}")" "$NOTE" >> "$OUT"

printf 'bench_speed: %s -> %s s  (appended to %s)\n' "$LABEL" "$mean" "${OUT#$ROOT/}"
