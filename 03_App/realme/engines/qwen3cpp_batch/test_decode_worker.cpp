// Model-free: the stream's decode worker hands back every sentence, in the
// order given, with the decoder's own samples; a failed decode is reported;
// stopping with work queued joins cleanly. Build with -fsanitize=thread too.
#include "qwen3_tts.h"
#define REALME_BATCH_NO_API
#include "realme_batch.inc"
#include <chrono>
#include <cstdio>
#include <random>
using namespace qwen3_tts;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);std::exit(1);}}while(0)
struct fake_decoder {
    int fail_at = -1, calls = 0;
    std::mt19937 rng{5};
    bool decode(const int32_t * codes, int32_t n_frames, std::vector<float> & out) {
        std::this_thread::sleep_for(std::chrono::microseconds(rng() % 3000));
        if (calls++ == fail_at) return false;
        out.assign((size_t) n_frames * 4, 0.0f);
        for (int32_t i = 0; i < n_frames * 4; ++i) out[i] = float(codes[(i / 4) * 16] + i % 4);
        return true;
    }
    std::string get_error() const { return "fake failure"; }
};
static std::vector<int32_t> codes_for(int id, int frames) {
    std::vector<int32_t> c((size_t) frames * 16);
    for (size_t i = 0; i < c.size(); ++i) c[i] = int32_t(id * 1000 + i);
    return c;
}
int main() {
    {   // order and content, collected while work is still being queued
        fake_decoder d;
        realme_batch::decode_worker_t<fake_decoder> w(d, 16);
        std::vector<int64_t> got;
        for (int id = 0; id < 40; ++id) {
            w.push(id, codes_for(id, id % 7));            // some with no frames
            realme_batch::decode_worker_t<fake_decoder>::done_item it;
            while (w.take(it, false)) {
                CHECK(it.ok);
                const int frames = int(it.id % 7);
                CHECK(it.audio.size() == size_t(frames) * 4);
                for (int i = 0; i < frames * 4; ++i) CHECK(it.audio[i] == float(it.id * 1000 + (i / 4) * 16 + i % 4));
                got.push_back(it.id);
            }
        }
        realme_batch::decode_worker_t<fake_decoder>::done_item it;
        while (w.take(it, true)) got.push_back(it.id);
        CHECK(got.size() == 40);
        for (int i = 0; i < 40; ++i) CHECK(got[i] == i);
        CHECK(!w.take(it, true));                       // nothing left: returns at once
    }
    {   // a failed decode is reported with the decoder's error, in its place
        fake_decoder d; d.fail_at = 3;
        realme_batch::decode_worker_t<fake_decoder> w(d, 16);
        for (int id = 0; id < 6; ++id) w.push(id, codes_for(id, 2));
        realme_batch::decode_worker_t<fake_decoder>::done_item it;
        int n = 0;
        while (w.take(it, true)) {
            CHECK(it.id == n);
            CHECK(it.ok == (n != 3));
            if (!it.ok) CHECK(it.error.find("fake failure") != std::string::npos);
            ++n;
        }
        CHECK(n == 6);
    }
    {   // destroyed with work queued: finishes it and joins
        fake_decoder d;
        auto * w = new realme_batch::decode_worker_t<fake_decoder>(d, 16);
        for (int id = 0; id < 10; ++id) w->push(id, codes_for(id, 3));
        delete w;
        CHECK(d.calls == 10);
    }
    fprintf(stderr, "PASS: decode worker order, samples, empty sentences, failure in place, join with work queued\n");
    return 0;
}
