// No models: verify buffer initialization, ownership, and allocation reuse.
#include "realme_batch_scratch.inc"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <new>

static size_t allocations = 0;
static size_t live_allocations = 0;
void * operator new(size_t size) {
    ++allocations;
    if (void * p = std::malloc(size ? size : 1)) { ++live_allocations; return p; }
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { if (p) --live_allocations; std::free(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }

using qwen3_tts::realme_batch_scratch;
static constexpr int H = 1024, V = 3072;

static void exercise(realme_batch_scratch & w, int batch, int step) {
    auto & in = w.reset(w.in, size_t(H) * 2 * batch);
    auto & pos = w.reset(w.pos, 2 * batch);
    for (float x : in) assert(x == 0.0f);
    for (int x : pos) assert(x == 0);
    for (size_t i = 0; i < in.size(); ++i) in[i] = float(i % 31) * 0.125f;
    for (int i = 0; i < batch; ++i) { pos[2*i] = 0; pos[2*i+1] = 1; }
    auto & code = w.reset(w.code, batch);
    auto & cpos = w.reset(w.cpos, batch, step + 1);
    for (int x : code) assert(x == 0);
    for (int x : cpos) assert(x == step + 1);
    for (int i = 0; i < batch; ++i) code[i] = (i + step) % 2048;
    auto & input = w.reset(w.step_in, 0);
    assert(input.empty());
    for (int b = 0; b < batch; ++b) {
        auto & e = w.reset(w.embedding, H);
        for (float x : e) assert(x == 0.0f);
        for (int h = 0; h < H; ++h) e[h] = float(h % 13);
        for (int cb = 1; cb < 16; ++cb)
            for (int h = 0; h < H; ++h) e[h] += float((h + cb + b) % 7) * 0.25f;
        input.insert(input.end(), e.begin(), e.end());
    }
    auto & hidden = w.reset(w.hidden, size_t(H) * batch);
    auto & logits = w.reset(w.logits, size_t(V) * batch);
    for (float x : hidden) assert(x == 0.0f);
    for (float x : logits) assert(x == 0.0f);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(i % 5) - 2.0f;
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = float(i % 19) * 0.5f;
}

template<class T> static void equal(const std::vector<T> & a, const std::vector<T> & b) {
    assert(a.size() == b.size());
    assert(a.empty() || std::memcmp(a.data(), b.data(), a.size()*sizeof(T)) == 0);
}

int main() {
    realme_batch_scratch reused(true), fresh(false);
    const int sizes[] = {8, 1, 4, 0, 8, 3, 2, 7, 1};
    for (int cycle = 0; cycle < 5; ++cycle) {
        for (int size : sizes) {
            exercise(reused, size, cycle);
            exercise(fresh, size, cycle);
            equal(reused.in,fresh.in); equal(reused.pos,fresh.pos);
            equal(reused.code,fresh.code); equal(reused.cpos,fresh.cpos);
            equal(reused.embedding,fresh.embedding); equal(reused.step_in,fresh.step_in);
            equal(reused.hidden,fresh.hidden); equal(reused.logits,fresh.logits);
        }
    }
    // All maximum shapes have been seen; no new host buffers should be needed.
    const auto before = allocations;
    for (int cycle = 0; cycle < 100; ++cycle)
        for (int size : sizes) exercise(reused, size, cycle % 14);
    assert(allocations == before);
    const auto control_before = allocations;
    for (int size : sizes) exercise(fresh, size, 1);
    assert(allocations > control_before);
    // Separate simultaneous calls own separate storage, and a reset clears it.
    realme_batch_scratch other(true);
    exercise(other, 8, 0);
    assert(other.in.data() != reused.in.data());
    const size_t bytes = (reused.in.capacity()+reused.step_in.capacity()+reused.embedding.capacity()
                       +reused.hidden.capacity()+reused.logits.capacity())*sizeof(float)
                       +(reused.pos.capacity()+reused.code.capacity()+reused.cpos.capacity())*sizeof(int32_t);
    assert(bytes < 1024*1024);
    const char * flag = std::getenv("QWEN3_TTS_SCRATCH_REUSE");
    assert(realme_batch_scratch::enabled() == (!flag || flag[0] != '0'));
    assert(realme_batch_scratch().reuse == realme_batch_scratch::enabled());
    const size_t live_before = live_allocations;
    { realme_batch_scratch temporary; exercise(temporary, 8, 0); }
    assert(live_allocations == live_before);
    std::printf("PASS: shrinking/refilling buffers match fresh bytes; 900 warmed cycles allocate zero buffers; control allocates; independent ownership; retained capacity %zu bytes\n", bytes);
}
