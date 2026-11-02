#pragma once
#include "realme_runtime.inc"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <utility>
#include <vector>
namespace qwen3_tts {
static float realme_topk_threshold(const float * logits, int count, int k,
                                  std::vector<float> & scratch) {
    // Keep the legacy comparator path for invalid numerical inputs as well.
    if (!realme_cpu_enabled("QWEN3_TTS_TOPK_SELECT") ||
        std::any_of(logits,logits+count,[](float x){ return std::isnan(x); })) {
        std::vector<std::pair<float,int>> scored(count);
        for (int i=0;i<count;++i) scored[i]={logits[i],i};
        std::partial_sort(scored.begin(),scored.begin()+k,scored.end(),
            [](const auto & a,const auto & b){ return a.first>b.first; });
        return scored[k-1].first;
    }
    scratch.assign(logits,logits+count);
    std::nth_element(scratch.begin(),scratch.begin()+k-1,scratch.end(),std::greater<float>());
    return scratch[k-1];
}
}
