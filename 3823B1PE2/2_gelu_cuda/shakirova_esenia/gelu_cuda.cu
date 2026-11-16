#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ float gelu_one(float x) {
    const float inner = x + 0.044715f * x * x * x;
    const float two_s = 1.5957691216057307f * inner;
    return x * __fdividef(1.0f, 1.0f + __expf(-two_s));
}

__global__ void gelu_packed(const float* __restrict__ in, float* __restrict__ out, int n4) {
    const float4* in4 = reinterpret_cast<const float4*>(in);
    float4* out4 = reinterpret_cast<float4*>(out);
    const int step = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += step) {
        const float4 v = __ldg(in4 + i);
        out4[i] = make_float4(gelu_one(v.x), gelu_one(v.y), gelu_one(v.z), gelu_one(v.w));
    }
}

__global__ void gelu_rest(const float* __restrict__ in, float* __restrict__ out, int n,
                         int offset) {
    const int i = offset + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = gelu_one(in[i]);
    }
}

struct GpuSession {
    float* dev_x = nullptr;
    float* dev_y = nullptr;
    const float* hooked_in = nullptr;
    float* hooked_out = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t cap = 0;
};

GpuSession& session() {
    static GpuSession s;
    return s;
}

void unhook(const float*& slot) {
    if (slot != nullptr) {
        cudaHostUnregister(const_cast<float*>(slot));
        slot = nullptr;
    }
}

void unhook_mut(float*& slot) {
    if (slot != nullptr) {
        cudaHostUnregister(slot);
        slot = nullptr;
    }
}

void reserve_device(std::size_t n) {
    GpuSession& s = session();
    if (s.stream == nullptr) {
        cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
    }
    if (s.cap >= n) {
        return;
    }
    if (s.dev_x != nullptr) {
        cudaFree(s.dev_x);
        cudaFree(s.dev_y);
    }
    const std::size_t bytes = n * sizeof(float);
    cudaMalloc(&s.dev_x, bytes);
    cudaMalloc(&s.dev_y, bytes);
    s.cap = n;
}

void hook_in(const float* ptr, std::size_t bytes) {
    GpuSession& s = session();
    if (s.hooked_in == ptr) {
        return;
    }
    unhook(s.hooked_in);
    if (cudaHostRegister(const_cast<float*>(ptr), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        s.hooked_in = ptr;
    }
}

void hook_out(float* ptr, std::size_t bytes) {
    GpuSession& s = session();
    if (s.hooked_out == ptr) {
        return;
    }
    unhook_mut(s.hooked_out);
    if (cudaHostRegister(ptr, bytes, cudaHostRegisterDefault) == cudaSuccess) {
        s.hooked_out = ptr;
    }
}

}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    reserve_device(static_cast<std::size_t>(n));
    GpuSession& s = session();
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    hook_in(input.data(), bytes);

    cudaMemcpyAsync(s.dev_x, input.data(), bytes, cudaMemcpyHostToDevice, s.stream);

    std::vector<float> result(static_cast<std::size_t>(n));

    const int n4 = n >> 2;
    if (n4 > 0) {
        int blocks = 1536;
        if (blocks > n4) {
            blocks = (n4 + kThreads - 1) / kThreads;
            if (blocks < 1) {
                blocks = 1;
            }
        }
        gelu_packed<<<blocks, kThreads, 0, s.stream>>>(s.dev_x, s.dev_y, n4);
    }
    if ((n & 3) != 0) {
        gelu_rest<<<1, 32, 0, s.stream>>>(s.dev_x, s.dev_y, n, n4 << 2);
    }

    hook_out(result.data(), bytes);
    cudaMemcpyAsync(result.data(), s.dev_y, bytes, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    return result;
}
