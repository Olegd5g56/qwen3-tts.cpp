#include "log.h"

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <mutex>
#include <string>
#include <unistd.h>

namespace qwen3_tts {

const char * const LVL_INFO  = "INFO ";
const char * const LVL_WARN  = "WARN ";
const char * const LVL_ERROR = "ERROR";

namespace {

inline bool color_enabled() {
    static const bool v = isatty(fileno(stderr)) && std::getenv("NO_COLOR") == nullptr;
    return v;
}

inline std::string ts() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    const auto t   = system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t, &tm);
    char buf[24];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03lld",
             tm.tm_hour, tm.tm_min, tm.tm_sec, (long long)ms.count());
    return buf;
}

inline const char * color(char tag) {
    if (!color_enabled()) return "";
    switch (tag) {
        case 'W': return "\033[33m";
        case 'E': return "\033[31m";
        default:  return "\033[2m";
    }
}
inline const char * reset() { return color_enabled() ? "\033[0m" : ""; }

void log_v(const char * level, const char * req_id, const char * fmt, va_list ap) {
    // One mutex around the whole emit so concurrent log calls don't shred each
    // other's lines into pieces.
    static std::mutex log_mu;
    std::lock_guard<std::mutex> lock(log_mu);
    fprintf(stderr, "%s%s %s%s", color(level[0]), ts().c_str(), level, reset());
    if (req_id && req_id[0]) fprintf(stderr, " [%s]", req_id);
    fputc(' ', stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
}

}  // namespace

void log_info(const char * fmt, ...) {
    va_list a; va_start(a, fmt); log_v(LVL_INFO, "", fmt, a); va_end(a);
}
void log_warn(const char * fmt, ...) {
    va_list a; va_start(a, fmt); log_v(LVL_WARN, "", fmt, a); va_end(a);
}
void log_error(const char * fmt, ...) {
    va_list a; va_start(a, fmt); log_v(LVL_ERROR, "", fmt, a); va_end(a);
}
void log_req(const char * req_id, const char * fmt, ...) {
    va_list a; va_start(a, fmt); log_v(LVL_INFO, req_id, fmt, a); va_end(a);
}
void log_req_warn(const char * req_id, const char * fmt, ...) {
    va_list a; va_start(a, fmt); log_v(LVL_WARN, req_id, fmt, a); va_end(a);
}
void log_at(const char * level, const char * req_id, const char * fmt, ...) {
    va_list a; va_start(a, fmt); log_v(level, req_id, fmt, a); va_end(a);
}

}  // namespace qwen3_tts
