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
# Two things this cannot do, learned the hard way:
#
#   * It cannot resolve a change worth less than ~100 ms. The seed is random,
#     so every run draws a different number of frames; a 73-character line
#     scattered 1.81-2.81 s across five runs of identical code. For a change
#     that only moves prefill, measure prefill directly with a fixed seed -
#     see optimization.md, "Per-voice prefill reuse".
#   * The commit it records goes "-dirty" as soon as the first row is appended,
#     because that dirties this history file. A run of several rows will show
#     the first clean and the rest dirty on the same code.
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
#   --voice NAME      voice from --voices-dir to use instead of the built-in
#                     benchmark voice. Recorded in the row: a different voice
#                     speaks at a different rate, so the same text becomes a
#                     different number of frames and a different number of
#                     seconds.
#   --voices-dir DIR  library to take --voice from (default: $TTS_VOICES_DIR)
#   --text FILE       input text       (default: benchmarks/bench_ru.txt)
#   --runs N          timed runs after the warm-up (default: 4)
#   --port N          (default: 8099)
#   --out FILE        history file     (default: benchmarks/speed.tsv)
#   --note TEXT       free-form note stored with the row
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD= LABEL= MODEL=${TTS_MODEL:-} VOCODER=${TTS_VOCODER:-} VOICE= VOICES_DIR=${TTS_VOICES_DIR:-}
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
[ -z "$VOICE" ] || [ -d "${VOICES_DIR:-}/$VOICE" ] || die "no voice '$VOICE' in '${VOICES_DIR:-<unset>}'"
command -v jq   >/dev/null || die "jq is required"
command -v curl >/dev/null || die "curl is required"

# The server is pointed at a directory holding exactly one voice. Two reasons:
# the background preload of a real library competes with the measurement, and a
# benchmark must not depend on a directory whose contents come and go.
#
# The default voice ships with the repo. It is 8.9 s of speech this model
# generated itself from benchmarks/voice/sample.txt, so the transcript matches
# the audio exactly - unlike examples/readme_clone_input.wav, whose 60 s do not
# match the 8 s transcript beside it and which drives generation to the token
# cap (known-issues.md). Without any reference at all the model is unconstrained
# and useless to time: the same text came out 717 frames one run and 496 the
# next, a 45% spread against ~5% with a voice.
ONEVOICE=$(mktemp -d); trap 'rm -rf "$ONEVOICE"' EXIT
if [ -n "$VOICE" ]; then
    cp -rL "$VOICES_DIR/$VOICE" "$ONEVOICE/$VOICE" || die "cannot copy voice $VOICE"
else
    VOICE=bench
    [ -f "$ROOT/benchmarks/voice/sample.mp3" ] || die "benchmarks/voice/sample.mp3 is missing"
    mkdir -p "$ONEVOICE/$VOICE"
    cp "$ROOT/benchmarks/voice/sample.mp3" "$ROOT/benchmarks/voice/sample.txt" "$ONEVOICE/$VOICE/" \
        || die "cannot stage the built-in benchmark voice"
fi

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
