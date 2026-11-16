#include "gelu_omp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <malloc.h>
#include <omp.h>

namespace {

void prefer_heap_reuse() {
    mallopt(M_MMAP_THRESHOLD, 1 << 29);
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_ARENA_MAX, 4);
}

void claim_len(std::vector<float>& v, std::size_t n) {
    v.reserve(n);
    struct Handle : std::vector<float> {
        static void set_len(std::vector<float>& v, std::size_t n) {
            auto& self = static_cast<Handle&>(v);
            self._M_impl._M_finish = self._M_impl._M_start + static_cast<std::ptrdiff_t>(n);
        }
    };
    Handle::set_len(v, n);
}

__attribute__((target("avx2,fma")))
void apply_gelu_span(const float* __restrict__ src, float* __restrict__ dst,
                     std::ptrdiff_t begin, std::ptrdiff_t end) {
    constexpr float kScale = 1.5957691216057307f;
    constexpr float kCubic = 0.07135481682360023f;
#pragma omp simd aligned(src, dst : 16)
    for (std::ptrdiff_t i = begin; i < end; ++i) {
        const float x = src[i];
        const float t = x * (kScale + kCubic * x * x);
        dst[i] = x / (1.0f + expf(-t));
    }
}

}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    static const int kInit = (prefer_heap_reuse(), 0);
    (void)kInit;

    const std::size_t n = input.size();
    std::vector<float> output;
    if (n == 0) {
        return output;
    }

    claim_len(output, n);
    const float* src = input.data();
    float* dst = output.data();
    const std::ptrdiff_t nn = static_cast<std::ptrdiff_t>(n);

#pragma omp parallel
    {
        const int threads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const std::ptrdiff_t chunk = (nn + threads - 1) / threads;
        const std::ptrdiff_t begin = std::min(nn, chunk * static_cast<std::ptrdiff_t>(tid));
        const std::ptrdiff_t end = std::min(nn, begin + chunk);
        apply_gelu_span(src, dst, begin, end);
    }

    return output;
}
