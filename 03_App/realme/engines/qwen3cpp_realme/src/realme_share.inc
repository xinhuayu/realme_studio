// realme_share.inc -- RealMe: an 8-bit product's rows shared out as threads
// ask for them, for processors with fast and slow cores.
//
// RealMe's 8-bit products (realme_q8mm_op, realme_q8_group_op) split the
// rows evenly between the threads: thread i takes the i-th share. Where
// every core is alike that is ideal. Intel Core 12th generation and Core
// Ultra (and Apple and many ARM chips) mix performance and efficiency cores;
// an even split leaves the fast cores waiting for the slow ones at the end of
// every product, hundreds of times a frame. Shared out, each thread takes the
// next small chunk of rows when it is free, so a fast core simply takes more.
//
// Each row is still computed whole by one thread with the same kernel, so the
// numbers are exactly the same whoever computes it. QWEN3_TTS_Q8_SHARE=1 on,
// =0 off; unset, on where the processor reports itself hybrid (CPUID 7.0 EDX
// bit 15), so computers with cores all alike keep the even split.
//
// The chunk counters live in the product node's own op_params, after GGML's
// custom-op header (zeroed when the node is made). Every thread of the graph
// calls a custom op once per compute; the last one to finish resets the
// counters, and GGML's barrier after the node orders that before the next
// compute of the same node.
#pragma once
#include "ggml.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace qwen3_tts {

static_assert(GGML_MAX_OP_PARAMS >= 40, "room for the share counters after the custom-op header");
static constexpr size_t REALME_SHARE_OFFSET = 32;   // after {fun, n_tasks, userdata}: 24 bytes on 64-bit

static bool realme_cpu_hybrid() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int r[4] = {0, 0, 0, 0};
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuidex(r, 7, 0);
    return ((r[3] >> 15) & 1) != 0;
#elif defined(__x86_64__) || defined(__i386__)
    unsigned a = 0, b = 0, c = 0, d = 0;
    if (__get_cpuid_max(0, nullptr) < 7) return false;
    __cpuid_count(7, 0, a, b, c, d);
    return ((d >> 15) & 1) != 0;
#else
    return false;
#endif
}

// The rule this rests on -- every graph thread calls a custom op once per
// compute -- is GGML's today, not a promise. So the first time sharing would
// be used, a small graph is computed twice with it (realme_share_selfcheck,
// in realme_q8_compute.inc): the same numbers as GGML's dot product and the
// counters back at zero, or sharing stays off. Decided while a graph is
// being built (realme_q8_eligible), never inside one.
static bool realme_share_selfcheck();
static int realme_share_state = -1;      // -1 undecided, 0 off, 1 on

static bool realme_q8_share_on() {
    if (realme_share_state >= 0) return realme_share_state == 1;
    const char * v = std::getenv("QWEN3_TTS_Q8_SHARE");
    const bool wanted = (v && v[0] == '1') || (!(v && v[0] == '0') && realme_cpu_hybrid());
    if (!wanted) { realme_share_state = 0; return false; }
    realme_share_state = 1;                // the check's own products share out
    const bool ok = realme_share_selfcheck();
    realme_share_state = ok ? 1 : 0;
    fprintf(stderr, "  [realme] 8-bit rows shared out between cores: %s\n",
            ok ? "on" : "off (self-check failed; even split)");
    return ok;
}

static inline int32_t * realme_share_counter(ggml_tensor * dst, int which) {
    return reinterpret_cast<int32_t *>(reinterpret_cast<char *>(dst->op_params) + REALME_SHARE_OFFSET) + which;
}
static inline int32_t realme_share_add(int32_t * p, int32_t v) {
#if defined(_MSC_VER)
    return (int32_t) _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(p), (long) v);
#else
    return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL);
#endif
}
static inline void realme_share_store(int32_t * p, int32_t v) {
#if defined(_MSC_VER)
    _InterlockedExchange(reinterpret_cast<volatile long *>(p), (long) v);
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

// Units [0, total) in chunks of about total / (8 x threads), each given to
// the thread that asks next: work(begin, end) for each chunk this thread takes.
template <class Work>
static void realme_share_out(ggml_tensor * dst, int nth, int64_t total, Work && work) {
    int32_t * claims = realme_share_counter(dst, 0);
    int32_t * done = realme_share_counter(dst, 1);
    const int64_t chunk = std::max<int64_t>(1, total / (8 * (int64_t) std::max(1, nth)));
    const int64_t chunks = (total + chunk - 1) / chunk;
    for (int64_t c; (c = realme_share_add(claims, 1)) < chunks;)
        work(c * chunk, std::min(total, (c + 1) * chunk));
    if (realme_share_add(done, 1) + 1 == nth) {
        realme_share_store(claims, 0);
        realme_share_store(done, 0);
    }
}

}  // namespace qwen3_tts
