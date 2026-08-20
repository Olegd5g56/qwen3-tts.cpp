#pragma once

// Shared fixture handling for the reference tests.
//
// Most of these tests compare against binary dumps produced by the original
// Python implementation, plus model files — none of which live in the repo
// (see .gitignore). On a machine without them every test failed, which made a
// red ctest run meaningless. They now report themselves as *skipped* instead:
// ctest maps the exit code below to SKIP via SKIP_RETURN_CODE.
//
// Model paths can be pointed somewhere else with QWEN3_TTS_TEST_MODEL and
// QWEN3_TTS_TEST_VOCODER, so a test that needs only a model (rather than a
// reference dump) can actually run.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace qwen3_tts_test {

// ctest's conventional "skipped" exit code, matching autotools.
constexpr int SKIP_EXIT_CODE = 77;

inline bool file_exists(const std::string & path) {
    struct stat st;
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Environment override for a model path, falling back to the built-in default.
inline std::string model_path_from_env(const char * env_name, const std::string & fallback) {
    const char * v = std::getenv(env_name);
    return (v && v[0]) ? std::string(v) : fallback;
}

// Returns true when every path is present. Otherwise prints what is missing
// and the caller should return SKIP_EXIT_CODE.
inline bool fixtures_present(const std::vector<std::string> & paths) {
    bool ok = true;
    for (const std::string & p : paths) {
        if (!file_exists(p)) {
            printf("  SKIP: fixture not found: %s\n", p.c_str());
            ok = false;
        }
    }
    if (!ok) {
        printf("\nThis test needs model files and/or reference dumps that are not\n"
               "part of the repository. Point QWEN3_TTS_TEST_MODEL /\n"
               "QWEN3_TTS_TEST_VOCODER at local ggufs, or regenerate the\n"
               "reference/ dumps, to run it.\n");
    }
    return ok;
}

} // namespace qwen3_tts_test
