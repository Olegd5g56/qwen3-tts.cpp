#include "op_profiler.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace qwen3_tts {

namespace {

using clk = std::chrono::high_resolution_clock;

struct op_stats {
    double  total_ms = 0.0;
    int64_t calls    = 0;
};

struct label_stats {
    // Keyed by "op name + output shape" so the same conv at different tower
    // depths shows up as a separate row.
    std::map<std::string, op_stats> ops;
    clk::time_point                 last;
    bool                            have_last = false;
};

std::map<std::string, label_stats> g_labels;

std::string node_key(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%-18s [%6lld,%5lld,%3lld]",
             ggml_op_name(t->op),
             (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
    return std::string(buf);
}

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        // Ask phase: request the post-compute callback for every node.
        return true;
    }
    auto &     st  = g_labels[(const char *) user_data];
    const auto now = clk::now();
    if (st.have_last) {
        auto & op = st.ops[node_key(t)];
        op.total_ms += std::chrono::duration<double, std::milli>(now - st.last).count();
        op.calls++;
    }
    st.last      = now;
    st.have_last = true;
    return true;
}

}  // namespace

bool op_profiler_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("QWEN3_TTS_PROFILE_OPS");
        return env && env[0] && env[0] != '0';
    }();
    return enabled;
}

void op_profiler_attach(ggml_backend_sched_t sched, const char * label) {
    if (!op_profiler_enabled() || !sched || !label) {
        return;
    }
    ggml_backend_sched_set_eval_callback(sched, eval_callback, (void *) label);
}

void op_profiler_report(const char * label) {
    if (!op_profiler_enabled() || !label) {
        return;
    }
    auto it = g_labels.find(label);
    if (it == g_labels.end()) {
        return;
    }

    std::vector<std::pair<std::string, op_stats>> rows(it->second.ops.begin(), it->second.ops.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto & a, const auto & b) { return a.second.total_ms > b.second.total_ms; });

    double total = 0.0;
    for (const auto & r : rows) {
        total += r.second.total_ms;
    }

    log_info("=== op profile [%s]: %.1f ms across %zu distinct nodes ===", label, total, rows.size());
    const size_t n_show = std::min<size_t>(rows.size(), 30);
    for (size_t i = 0; i < n_show; ++i) {
        const auto & r = rows[i];
        log_info("  %8.1f ms (%4.1f%%)  n=%-7lld %s",
                 r.second.total_ms, total > 0 ? 100.0 * r.second.total_ms / total : 0.0,
                 (long long) r.second.calls, r.first.c_str());
    }

    g_labels.erase(it);
}

}  // namespace qwen3_tts
