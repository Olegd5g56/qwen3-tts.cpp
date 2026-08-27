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
#   * A FIXED SEED, which is the single thing that makes this measurable. The
#     headline is whole-request wall time, and wall time is dominated by how
#     much audio the model chose to generate. With a random seed every run
#     draws a different length and the number wanders far more than any change
#     worth measuring. Measured 2026-08-27, ROCm 6800XT, bench_ru.txt, 12 runs
#     each:
#
#         random seed   audio 38.72-44.88 s (14.7%)   wall 9.81-11.01 s (11.5%)
#         fixed seed    audio 39.44 s every run (0%)  wall  9.61-10.01 s (4.2%)
#
#     The audio length is bit-identical across runs, so what is left is the
#     machine, not the draw.
#   * Nine timed requests, all recorded, and the MEDIAN reported - not the
#     mean. On this machine roughly every fourth request runs ~300 ms slower
#     (~3%), entirely inside the generation loop, on a period of about 38 s.
#     A mean of four gives that hiccup a quarter of the weight; a median of
#     nine ignores it.
#   * The audio duration of every run is recorded and checked. If it moves
#     while the seed is pinned, the run is not reproducible and the wall times
#     in it compare nothing - the script says so instead of averaging them.
#   * One voice in the voices directory. The server preloads the whole library
#     in the background and 198 voices compete with what is being measured.
#
# Comparing two rows, which is where this gets misused:
#
#   * Same backend, same model, a change that does not touch numerics: compare
#     median_s directly. The audio_s column must match; if it does not, the
#     change moved the token stream and seconds are not comparable.
#   * Different backend, different weight type, anything that changes the
#     tokens: the audio lengths differ by construction, so seconds are
#     meaningless. Compare ms_per_frame, which is whole-request wall time
#     divided by the frames it produced - still an end-to-end number, just
#     normalised by how much came out. It is NOT the server's own generation
#     ms/frame, which is not comparable across backends because the vocoder is
#     overlapped into generation on CUDA/ROCm/Metal and not on Vulkan/CPU.
#
# What this still cannot do:
#
#   * Resolve a change worth less than ~1% of the request. The floor is now
#     the ~3% hiccup above, and the median only hides it, it does not remove
#     it. For a change that only moves prefill, measure prefill directly -
#     see optimization.md, "Per-voice prefill reuse".
#
# Usage:
#   scripts/bench_speed.sh --build build-cuda --label "CUDA 1660S" [options]
#
#   --build DIR       build directory holding qwen3-tts-server
#   --docker IMAGE    benchmark a built image instead of a build directory.
#                     The deployed artefact ships its own userspace GPU driver -
#                     the Vulkan image carries Debian's mesa, not the host's -
#                     so a host build and the image it is built from are two
#                     different configurations and can differ. Exactly one of
#                     --build / --docker is required.
#   --label NAME      how this configuration is named in the log (required)
#   --env K=V         extra environment for the server (repeatable),
#                     e.g. --env CUDA_VISIBLE_DEVICES=0
#   --unset-env K     remove a variable the image's own ENV sets (repeatable,
#                     --docker only). Needed because ggml's Vulkan switches test
#                     for a variable's PRESENCE, not its value: setting
#                     GGML_VK_DISABLE_MULTI_ADD=0 disables multi-add just as
#                     surely as =1 does, so the only way to measure the image
#                     without its own workaround is to unset it.
#   --model PATH      talker gguf      (default: $TTS_MODEL)
#   --vocoder PATH    vocoder gguf     (default: $TTS_VOCODER)
#   --voice NAME      voice from --voices-dir to use instead of the built-in
#                     benchmark voice. Recorded in the row: a different voice
#                     speaks at a different rate, so the same text becomes a
#                     different number of frames and a different number of
#                     seconds.
#   --voices-dir DIR  library to take --voice from (default: $TTS_VOICES_DIR)
#   --text FILE       input text       (default: benchmarks/bench_ru.txt)
#   --runs N          timed runs after the warm-up (default: 9)
#   --seed N          sampling seed, pinned so every run generates the same
#                     audio (default: 1234). "--seed random" restores the old
#                     unpinned behaviour, which is only useful for measuring
#                     the spread itself.
#   --port N          (default: 8099)
#   --out FILE        history file     (default: benchmarks/speed.tsv)
#   --note TEXT       free-form note stored with the row
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD= LABEL= MODEL=${TTS_MODEL:-} VOCODER=${TTS_VOCODER:-} VOICE= VOICES_DIR=${TTS_VOICES_DIR:-}
TEXT="$ROOT/benchmarks/bench_ru.txt" RUNS=9 PORT=8099 OUT="$ROOT/benchmarks/speed.tsv" NOTE=
SEED=1234 DOCKER=
declare -a UNSET_ENV=()
declare -a EXTRA_ENV=()

while [ $# -gt 0 ]; do
    case "$1" in
        --build)       BUILD=$2; shift 2 ;;
        --docker)      DOCKER=$2; shift 2 ;;
        --unset-env)   UNSET_ENV+=("$2"); shift 2 ;;
        --label)       LABEL=$2; shift 2 ;;
        --env)         EXTRA_ENV+=("$2"); shift 2 ;;
        --model)       MODEL=$2; shift 2 ;;
        --vocoder)     VOCODER=$2; shift 2 ;;
        --voice)       VOICE=$2; shift 2 ;;
        --voices-dir)  VOICES_DIR=$2; shift 2 ;;
        --text)        TEXT=$2; shift 2 ;;
        --runs)        RUNS=$2; shift 2 ;;
        --seed)        SEED=$2; shift 2 ;;
        --port)        PORT=$2; shift 2 ;;
        --out)         OUT=$2; shift 2 ;;
        --note)        NOTE=$2; shift 2 ;;
        -h|--help)     sed -n '2,84p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

die() { echo "bench_speed: $*" >&2; exit 1; }
[ -n "$BUILD" ] || [ -n "$DOCKER" ] || die "one of --build / --docker is required"
[ -z "$BUILD" ] || [ -z "$DOCKER" ] || die "--build and --docker are mutually exclusive"
[ -n "$LABEL" ]  || die "--label is required"
[ ${#UNSET_ENV[@]} -eq 0 ] || [ -n "$DOCKER" ] || die "--unset-env only applies to --docker"
if [ -n "$BUILD" ]; then
    [ -x "$BUILD/qwen3-tts-server" ] || die "no qwen3-tts-server in $BUILD"
else
    command -v docker >/dev/null || die "docker is required for --docker"
    docker image inspect "$DOCKER" >/dev/null 2>&1 || die "no such image: $DOCKER"
fi
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
# the audio exactly. That property is the point: a reference paired with a
# transcript it does not match drives generation to the token cap, which is how
# examples/readme_clone_input.wav wasted an afternoon before it was deleted
# (known-issues.md #23). Without any reference at all the model is
# unconstrained and useless to time: the same text came out 717 frames one run
# and 496 the next, a 45% spread against ~5% with a voice.
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
# The history file is excluded from the dirty check: appending a row dirties it,
# so without this the first row of a session reads clean and every later one
# reads "-dirty" on identical code.
OUT_REL=${OUT#$ROOT/}
git -C "$ROOT" diff --quiet -- . ":(exclude)$OUT_REL" 2>/dev/null || COMMIT="$COMMIT-dirty"
if [ -n "$BUILD" ]; then
    NATIVE=$(cache GGML_NATIVE); TIMING=$(cache QWEN3_TTS_TIMING)
else
    # An image has no CMakeCache to read. Its identity is the image digest, and
    # that is what has to be in the row: two images built from the same commit
    # can still differ, because the base image's packages moved under them.
    NATIVE="img:$(docker image inspect -f '{{.Id}}' "$DOCKER" | cut -c8-19)"; TIMING=-
fi
# Everything passed with --env goes in the row. A device selection or a backend
# workaround changes the answer exactly as much as a build flag does, and a row
# that does not say which one was set cannot be compared to anything.
ENV_ID=$( [ ${#EXTRA_ENV[@]} -gt 0 ] && (IFS=' '; echo "${EXTRA_ENV[*]}") || echo '-' )
for u in "${UNSET_ENV[@]}"; do
    [ "$ENV_ID" = '-' ] && ENV_ID="unset:$u" || ENV_ID="$ENV_ID unset:$u"
done
TEXT_ID="$(basename "$TEXT"):$(wc -c < "$TEXT" | tr -d ' ')B"

if [ -n "$BUILD" ]; then
    "$BUILD/qwen3-tts-server" --help >/dev/null 2>&1  # fail fast on a broken binary
    env "${EXTRA_ENV[@]}" \
        TTS_MODEL="$MODEL" TTS_VOCODER="$VOCODER" TTS_VOICES_DIR="$ONEVOICE" \
        TTS_PORT="$PORT" TTS_HOST=127.0.0.1 TTS_VERBOSE=1 TTS_IDLE_TIMEOUT=3600 \
        "$BUILD/qwen3-tts-server" > "$ONEVOICE/server.log" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null; rm -rf "$ONEVOICE"' EXIT
    server_alive() { kill -0 $SERVER 2>/dev/null; }
    server_log()   { cat "$ONEVOICE/server.log"; }
else
    # Both gguf files are mounted by their own directory, read-only, and the
    # staged one-voice library likewise. --device /dev/dri --group-add video is
    # what the image's own header documents for GPU access.
    CNAME="bench-speed-$PORT-$$"
    docker_env=()
    for e in "${EXTRA_ENV[@]}"; do docker_env+=(-e "$e"); done
    # docker has no way to remove a variable the image's own ENV set - -e K=
    # leaves it present and empty, which getenv() still sees. So when something
    # has to be unset, the entrypoint becomes a shell that unsets it and execs
    # the server.
    docker_entry=(); docker_cmd=()
    if [ ${#UNSET_ENV[@]} -gt 0 ]; then
        unset_script=""
        for u in "${UNSET_ENV[@]}"; do unset_script+="unset $u; "; done
        docker_entry=(--entrypoint /bin/sh)
        docker_cmd=(-c "${unset_script}exec /opt/qwen3-tts/bin/qwen3-tts-server")
    fi
    docker run -d --rm --name "$CNAME" \
        --device /dev/dri --group-add video \
        -p "127.0.0.1:$PORT:8081" \
        -v "$(cd "$(dirname "$MODEL")" && pwd):/m:ro" \
        -v "$(cd "$(dirname "$VOCODER")" && pwd):/v:ro" \
        -v "$ONEVOICE:/voices:ro" \
        -e TTS_MODEL="/m/$(basename "$MODEL")" -e TTS_VOCODER="/v/$(basename "$VOCODER")" \
        -e TTS_VOICES_DIR=/voices -e TTS_HOST=0.0.0.0 -e TTS_PORT=8081 \
        -e TTS_VERBOSE=1 -e TTS_IDLE_TIMEOUT=3600 \
        "${docker_env[@]}" "${docker_entry[@]}" "$DOCKER" "${docker_cmd[@]}" \
        >/dev/null || die "docker run failed"
    trap 'docker rm -f "$CNAME" >/dev/null 2>&1; rm -rf "$ONEVOICE"' EXIT
    server_alive() { [ -n "$(docker ps -q -f name="^$CNAME\$")" ]; }
    server_log()   { docker logs "$CNAME" 2>&1; }
fi

for _ in $(seq 1 240); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    server_alive || { server_log >&2; die "server exited"; }
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || die "server never became healthy"

# The seed rides in the request body. It is not part of the OpenAI schema, but
# the server has always accepted it, and pinning it is what makes wall time
# mean anything - see the protocol note at the top.
request() {
    if [ "$SEED" = random ]; then
        jq -n --rawfile input "$TEXT" \
            '{input: $input, voice: "'"$VOICE"'", response_format: "wav"}'
    else
        jq -n --rawfile input "$TEXT" --argjson seed "$SEED" \
            '{input: $input, voice: "'"$VOICE"'", response_format: "wav", seed: $seed}'
    fi \
    | curl -s -m 900 -o /dev/null -X POST "http://127.0.0.1:$PORT/v1/audio/speech" \
        -H 'Content-Type: application/json' -d @-
}

# The server logs one line per finished request; the last one is ours. Duration
# comes from there rather than from the returned bytes so that a change to the
# container or the encoder cannot quietly move it.
last_log_field() {  # $1 = regex with one capture group
    server_log | grep -a 'ok .*audio' | tail -1 | sed -nE "s/.*$1.*/\\1/p"
}

printf 'bench_speed: %s  commit=%s voice=%s seed=%s native=%s timing=%s env=%s\n' \
    "$LABEL" "$COMMIT" "$VOICE" "$SEED" "${NATIVE:-?}" "${TIMING:-?}" "$ENV_ID"
request  # warm-up, discarded: it costs about double
times=() durations=() prefills=()
for i in $(seq 1 "$RUNS"); do
    start=$(date +%s.%N); request; end=$(date +%s.%N)
    t=$(LC_ALL=C awk "BEGIN{printf \"%.2f\", $end - $start}")
    dur=$(last_log_field 'ok ([0-9.]+)s audio')
    pre=$(last_log_field 'prefill=([0-9]+)ms')
    times+=("$t"); durations+=("${dur:-?}"); prefills+=("${pre:-?}")
    printf '  run %d: %s s wall, %s s audio\n' "$i" "$t" "${dur:-?}"
done

med() { printf '%s\n' "$@" | LC_ALL=C sort -n | awk '{v[NR]=$1}END{printf "%.2f", (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}'; }
median=$(med "${times[@]}")
prefill_ms=$(med "${prefills[@]}")

# A run with no log line behind it is not a slow measurement, it is a failed
# request - the server died, the container was removed, the port moved. Writing
# that as a row puts debris in a history whose whole value is that its rows are
# comparable. Refuse instead: an aborted run leaves nothing behind.
for d in "${durations[@]}"; do
    [ "$d" = '?' ] && die "a request produced no server log line - nothing was measured, no row written"
done

# Every run must have produced the same audio, or the seconds above are timing
# different amounts of work and their median means nothing. This is the check
# the old protocol did not have, and it is why it could not resolve anything.
uniq_dur=$(printf '%s\n' "${durations[@]}" | sort -u | tr '\n' ' ' | sed 's/ $//')
AUDIO_S=$uniq_dur
if [ "$(printf '%s\n' "${durations[@]}" | sort -u | wc -l)" -gt 1 ]; then
    AUDIO_S="VARIES:$(printf '%s\n' "${durations[@]}" | sort -u | tr '\n' ',' | sed 's/,$//')"
    printf 'bench_speed: WARNING - audio length varied across runs (%s).\n' "$uniq_dur" >&2
    printf '             The median wall time compares nothing. With --seed pinned this\n' >&2
    printf '             means generation is not reproducible on this build.\n' >&2
    NOTE="${NOTE:+$NOTE; }audio length varied, wall time not comparable"
fi

# Whole-request wall time per frame of audio produced: the same end-to-end
# number, divided by how much came out. Use it - and only it - when two rows
# have different audio lengths. Computed per run and then medianed, so that a
# run's seconds are always divided by that same run's frames.
mspfs=()
for i in "${!times[@]}"; do
    d=${durations[$i]}
    case "$d" in
        ''|'?') continue ;;
    esac
    mspfs+=("$(LC_ALL=C awk -v t="${times[$i]}" -v d="$d" \
        'BEGIN{ if (d+0 > 0) printf "%.3f", (t*1000)/(d*12) }')")
done
MSPF='?'
[ ${#mspfs[@]} -gt 0 ] && MSPF=$(printf '%s\n' "${mspfs[@]}" | LC_ALL=C sort -n \
    | awk '{v[NR]=$1}END{printf "%.3f", (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')

HEADER=$'date\tcommit\tlabel\tmodel\ttext\tvoice\tseed\tnative\ttiming\tenv\tmedian_s\taudio_s\tms_per_frame\tprefill_ms\truns_s\tnote'
if [ -s "$OUT" ]; then
    # A history written by an older protocol has different columns and its
    # numbers were produced a different way. Appending to it would make a file
    # that reads like one series and is two.
    [ "$(head -1 "$OUT")" = "$HEADER" ] || \
        die "$OUT was written by an older protocol (different columns). Move it aside; this run would corrupt the series."
else
    printf '%s\n' "$HEADER" > "$OUT"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%MZ)" "$COMMIT" "$LABEL" "$(basename "$MODEL")" "$TEXT_ID" "$VOICE" \
    "$SEED" "${NATIVE:-?}" "${TIMING:-?}" "$ENV_ID" "$median" "$AUDIO_S" "$MSPF" "$prefill_ms" \
    "$(IFS=,; echo "${times[*]}")" "$NOTE" >> "$OUT"

printf 'bench_speed: %s -> %s s median of %s  (%s s audio, %s ms/frame)  appended to %s\n' \
    "$LABEL" "$median" "$RUNS" "$AUDIO_S" "$MSPF" "${OUT#$ROOT/}"
