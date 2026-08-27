#include "op_profiler.h"
#include "env_config.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// --- numeric probe (QWEN3_TTS_PROBE_NUM=1) -------------------------------
//
// Records, per graph node, the largest |value| its output ever carried and how
// many non-finite values it produced. It exists to answer one question: do
// this model's activations leave the F16 range (max 65504) before they reach a
// matmul whose weights are F16? With F16 weights ggml converts the activation
// side to F16 too (ggml-cpu.c: vec_dot_type of GGML_TYPE_F16 is F16), so an
// activation above 65504 becomes inf there and nowhere else. Q8_0 weights
// cannot hit this - their activation side is block-scaled int8. See
// docs/known-issues.md #16.

struct num_stats {
    double  max_abs     = 0.0;
    int64_t n_nonfinite = 0;
    int64_t calls       = 0;
};

std::map<std::string, num_stats> g_num;
std::vector<uint8_t>             g_num_buf;
bool                             g_num_first_bad = false;

std::string num_key(const struct ggml_tensor * t) {
    char buf[320];
    snprintf(buf, sizeof(buf), "%-18s %-24s [%6lld,%5lld,%3lld]",
             ggml_op_name(t->op), t->name[0] ? t->name : "(unnamed)",
             (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
    return std::string(buf);
}

void scan_node(struct ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        return;
    }
    // Two sources of legitimate huge/non-finite values that would otherwise
    // drown the table: attention masks carry -FLT_MAX or -inf by design, and a
    // KV-cache view covers slots that have not been written yet.
    if (t->op == GGML_OP_DIAG_MASK_INF) {
        return;
    }
    if (strstr(t->name, "cache") != nullptr || strstr(t->name, "mask") != nullptr) {
        return;
    }
    // Views can be strided; reading nbytes then indexing linearly would walk
    // over neighbouring data, so only scan tensors laid out densely.
    if (!ggml_is_contiguous(t)) {
        return;
    }
    const size_t nb = ggml_nbytes(t);
    if (nb == 0 || nb > (size_t) 512 * 1024 * 1024) {
        return;
    }
    g_num_buf.resize(nb);
    ggml_backend_tensor_get(t, g_num_buf.data(), 0, nb);

    const int64_t n       = ggml_nelements(t);
    double        max_abs = 0.0;
    int64_t       bad     = 0;
    for (int64_t i = 0; i < n; ++i) {
        const float v = (t->type == GGML_TYPE_F32)
                            ? ((const float *) g_num_buf.data())[i]
                            : ggml_fp16_to_fp32(((const ggml_fp16_t *) g_num_buf.data())[i]);
        if (!std::isfinite(v)) {
            bad++;
            continue;
        }
        const double a = std::fabs((double) v);
        // Mask sentinels (-FLT_MAX) ride along inside otherwise ordinary
        // tensors after an add; they say nothing about activation scale.
        if (a > 1e30) {
            continue;
        }
        if (a > max_abs) {
            max_abs = a;
        }
    }

    auto & st = g_num[num_key(t)];
    st.calls++;
    st.n_nonfinite += bad;
    if (max_abs > st.max_abs) {
        st.max_abs = max_abs;
    }

    // The first node to go non-finite is the whole answer, so say it loudly
    // and once - after this the poison spreads and every row looks guilty.
    if (bad > 0 && !g_num_first_bad) {
        g_num_first_bad = true;
        log_warn("[probe] FIRST non-finite output: %s  (%lld of %lld values)",
                 num_key(t).c_str(), (long long) bad, (long long) n);
    }
}

std::string node_key(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%-18s [%6lld,%5lld,%3lld]",
             ggml_op_name(t->op),
             (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
    return std::string(buf);
}

bool num_probe_on();

bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) {
        // Ask phase: request the post-compute callback for every node.
        return true;
    }
    if (num_probe_on()) {
        scan_node(t);
    }
    if (!op_profiler_enabled()) {
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

bool num_probe_on() {
    return qwen3_tts::env().probe_num;
}

}  // namespace

bool op_profiler_enabled() {
    return qwen3_tts::env().profile_ops;
}

void op_profiler_attach(ggml_backend_sched_t sched, const char * label) {
    if ((!op_profiler_enabled() && !num_probe_on()) || !sched || !label) {
        return;
    }
    ggml_backend_sched_set_eval_callback(sched, eval_callback, (void *) label);
}

void op_profiler_report(const char * label) {
    if (num_probe_on() && !g_num.empty()) {
        std::vector<std::pair<std::string, num_stats>> nrows(g_num.begin(), g_num.end());
        std::sort(nrows.begin(), nrows.end(),
                  [](const auto & a, const auto & b) { return a.second.max_abs > b.second.max_abs; });

        int64_t total_bad = 0;
        for (const auto & r : nrows) {
            total_bad += r.second.n_nonfinite;
        }
        log_info("=== numeric probe [%s]: %zu nodes, %lld non-finite values, F16 max is 65504 ===",
                 label, nrows.size(), (long long) total_bad);
        // Always show every node that cannot survive a cast to F16, however
        // many that is - that list is the actionable part of the report.
        size_t n_over = 0;
        while (n_over < nrows.size() && nrows[n_over].second.max_abs > 65504.0) {
            n_over++;
        }
        const int    top   = qwen3_tts::env().probe_top;
        const size_t n_top = top > 0 ? (size_t) top : 25;
        const size_t n_show = std::min<size_t>(nrows.size(), std::max<size_t>(n_over, n_top));
        for (size_t i = 0; i < n_show; ++i) {
            const auto & r = nrows[i];
            log_info("  max|x|=%12.1f  nonfinite=%-8lld n=%-6lld %s",
                     r.second.max_abs, (long long) r.second.n_nonfinite,
                     (long long) r.second.calls, r.first.c_str());
        }
        g_num.clear();
        g_num_first_bad = false;
    }

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
