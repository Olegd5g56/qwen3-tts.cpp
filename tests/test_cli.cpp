// test_cli.cpp — end-to-end smoke test for the qwen3-tts-cli binary.
//
// Everything else in ctest links the libraries directly. Nothing ran the CLI,
// so its argument parsing, format dispatch and file writing had no automated
// cover at all — scripts/run_all_tests.sh nominally did it and had been
// silently skipping itself for months (known-issues.md). This replaces the one
// idea in that script worth keeping.
//
// It is a smoke test on purpose: it asserts that each path produces a playable
// file of a sane length, not what the audio says. Numerics belong to the tests
// that can compare against a reference.
//
// Needs a model: QWEN3_TTS_TEST_MODEL and QWEN3_TTS_TEST_VOCODER. Without them
// it exits 77 (ctest "skipped"), like the other model-dependent tests. The
// reference clip and transcript ship with the repo.

#include "test_fixtures.h"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

std::string cli;      // path to the binary under test
std::string model;
std::string vocoder;
std::string outdir;
int failures = 0;

std::string quote(const std::string & s) { return "'" + s + "'"; }

// Runs the CLI with the given trailing arguments. Output goes to a log file so
// a failure can print it; a passing test stays quiet.
int run_cli(const std::string & args, const std::string & logfile) {
    const std::string cmd = quote(cli) +
        " --model " + quote(model) + " --vocoder " + quote(vocoder) +
        " " + args + " > " + quote(logfile) + " 2>&1";
    return std::system(cmd.c_str());
}

void dump_log(const std::string & logfile) {
    FILE * f = fopen(logfile.c_str(), "rb");
    if (!f) return;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);
    printf("     --- cli output ---\n%s\n", buf);
}

std::vector<unsigned char> read_file(const std::string & path) {
    std::vector<unsigned char> data;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return data;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) data.insert(data.end(), buf, buf + n);
    fclose(f);
    return data;
}

uint32_t le32(const unsigned char * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}
uint16_t le16(const unsigned char * p) { return (uint16_t) ((uint32_t) p[0] | ((uint32_t) p[1] << 8)); }

// Walks the RIFF chunks rather than assuming a 44-byte header, so an encoder
// that adds a LIST chunk does not read as a broken file.
bool wav_seconds(const std::vector<unsigned char> & d, double & seconds, uint32_t & rate) {
    if (d.size() < 12 || memcmp(d.data(), "RIFF", 4) != 0 || memcmp(d.data() + 8, "WAVE", 4) != 0) {
        return false;
    }
    uint32_t byte_rate = 0;
    size_t pos = 12;
    while (pos + 8 <= d.size()) {
        const uint32_t size = le32(&d[pos + 4]);
        if (memcmp(&d[pos], "fmt ", 4) == 0 && size >= 16 && pos + 8 + 16 <= d.size()) {
            rate      = le32(&d[pos + 8 + 4]);
            byte_rate = le32(&d[pos + 8 + 8]);
        } else if (memcmp(&d[pos], "data", 4) == 0) {
            if (byte_rate == 0) return false;
            const uint32_t avail = (uint32_t) (d.size() - (pos + 8));
            seconds = (double) (size < avail ? size : avail) / (double) byte_rate;
            return true;
        }
        pos += 8 + size + (size & 1);   // chunks are word-aligned
    }
    return false;
}

// One case: run the CLI, expect a WAV of at least min_seconds.
void expect_wav(const char * what, const std::string & args, const std::string & out,
                double min_seconds) {
    const std::string log = outdir + "/cli.log";
    const int rc = run_cli(args + " --output " + quote(out), log);
    if (rc != 0) {
        printf("FAIL %s: cli exited %d\n", what, rc);
        dump_log(log);
        failures++;
        return;
    }
    double seconds = 0.0;
    uint32_t rate = 0;
    const std::vector<unsigned char> d = read_file(out);
    if (!wav_seconds(d, seconds, rate)) {
        printf("FAIL %s: not a readable WAV (%zu bytes)\n", what, d.size());
        failures++;
        return;
    }
    if (rate != 24000) {
        printf("FAIL %s: sample rate is %u, expected 24000\n", what, rate);
        failures++;
        return;
    }
    if (seconds < min_seconds) {
        printf("FAIL %s: %.2f s of audio, expected at least %.2f\n", what, seconds, min_seconds);
        failures++;
        return;
    }
    printf("ok   %-26s %.2f s\n", what, seconds);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path to qwen3-tts-cli>\n", argv[0]);
        return 2;
    }
    cli     = argv[1];
    model   = qwen3_tts_test::model_path_from_env("QWEN3_TTS_TEST_MODEL", "");
    vocoder = qwen3_tts_test::model_path_from_env("QWEN3_TTS_TEST_VOCODER", "");
    if (model.empty() || vocoder.empty()) {
        printf("  SKIP: set QWEN3_TTS_TEST_MODEL and QWEN3_TTS_TEST_VOCODER to run the CLI test\n");
        return qwen3_tts_test::SKIP_EXIT_CODE;
    }
    if (!qwen3_tts_test::fixtures_present({model, vocoder, "benchmarks/voice/sample.mp3",
                                           "benchmarks/voice/sample.txt"})) {
        return qwen3_tts_test::SKIP_EXIT_CODE;
    }

    outdir = "test_output_cli";
    std::system(("rm -rf " + quote(outdir) + " && mkdir -p " + quote(outdir)).c_str());

    const std::string seed = " --seed 1234";

    expect_wav("default voice", "--text 'A short line of speech.'" + seed,
               outdir + "/plain.wav", 0.5);

    // Digits are their own path: the frame budget counts a digit as roughly
    // four letters, and getting that wrong truncates the request
    // (known-issues.md #12).
    expect_wav("digits", "--text 'In 2024 the total was 1536 units.'" + seed,
               outdir + "/digits.wav", 1.0);

    expect_wav("punctuation", "--text 'Wait - really? Yes, really!'" + seed,
               outdir + "/punct.wav", 0.8);

    // ICL cloning through the CLI: clip plus its matching transcript.
    {
        std::vector<unsigned char> t = read_file("benchmarks/voice/sample.txt");
        std::string transcript((const char *) t.data(), t.size());
        while (!transcript.empty() && (transcript.back() == '\n' || transcript.back() == '\r')) {
            transcript.pop_back();
        }
        expect_wav("voice cloning (ICL)",
                   "--reference 'benchmarks/voice/sample.mp3' --ref-text " + quote(transcript) +
                   " --text 'Cloned from the reference clip.'" + seed,
                   outdir + "/clone.wav", 0.8);
    }

    // The CLI's own streaming path, which decodes while it generates.
    expect_wav("streaming batches", "--text 'Streamed while it generates.' --streaming-batch-size 16" + seed,
               outdir + "/stream.wav", 0.5);

    // A pinned seed must reproduce exactly - this is the property every
    // benchmark and every parity test in the repo leans on.
    {
        expect_wav("seeded run A", "--text 'Determinism check.'" + seed,
                   outdir + "/seed_a.wav", 0.4);
        expect_wav("seeded run B", "--text 'Determinism check.'" + seed,
                   outdir + "/seed_b.wav", 0.4);
        const std::vector<unsigned char> a = read_file(outdir + "/seed_a.wav");
        const std::vector<unsigned char> b = read_file(outdir + "/seed_b.wav");
        if (a.empty() || a != b) {
            printf("FAIL same seed produced different audio (%zu vs %zu bytes)\n", a.size(), b.size());
            failures++;
        } else {
            printf("ok   %-26s byte-identical\n", "same seed twice");
        }
    }

    // Format dispatch is by extension. MP3 starts with an ID3 tag or a frame
    // sync; Ogg with "OggS". Getting a WAV here would mean the extension was
    // ignored, which is the failure worth catching.
    {
        const std::string log = outdir + "/cli.log";
        struct { const char * name; const char * ext; const char * magic; size_t n; } cases[] = {
            { "mp3 output",  "mp3",  "ID3", 3 },
            { "opus output", "opus", "OggS", 4 },
        };
        for (const auto & c : cases) {
            const std::string out = outdir + "/fmt." + c.ext;
            const int rc = run_cli("--text 'Format check.'" + seed + " --output " + quote(out), log);
            const std::vector<unsigned char> d = read_file(out);
            bool ok = rc == 0 && d.size() > 512 && memcmp(d.data(), c.magic, c.n) == 0;
            if (!ok && rc == 0 && c.n == 3 && d.size() > 2 && d[0] == 0xFF && (d[1] & 0xE0) == 0xE0) {
                ok = true;   // a bare MPEG frame sync, no ID3 tag
            }
            if (!ok) {
                printf("FAIL %s: rc=%d, %zu bytes\n", c.name, rc, d.size());
                if (rc != 0) dump_log(log);
                failures++;
            } else {
                printf("ok   %-26s %zu bytes\n", c.name, d.size());
            }
        }
    }

    // An argument it does not know must be refused, not ignored. This is the
    // whole reason a caller can trust the flags above.
    {
        const std::string log = outdir + "/cli.log";
        const int rc = run_cli("--text 'x' --no-such-flag --output " + quote(outdir + "/never.wav"), log);
        if (rc == 0) {
            printf("FAIL unknown argument was accepted\n");
            failures++;
        } else {
            printf("ok   %-26s refused\n", "unknown argument");
        }
    }

    if (failures) {
        printf("\nFAIL: %d check(s) failed\n", failures);
        return 1;
    }
    printf("\nPASS\n");
    return 0;
}
