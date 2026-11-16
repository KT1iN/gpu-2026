#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int kTx = 32;
constexpr int kTy = 8;
constexpr int kRows = 4;
constexpr int kCols = 4;

__global__ void gemm_coalesced(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c, int n) {
    const int col = (blockIdx.x * kTx + threadIdx.x) * kCols;
    const int row = (blockIdx.y * kTy + threadIdx.y) * kRows;

    float acc[kRows][kCols];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
#pragma unroll
        for (int j = 0; j < kCols; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const float* ap[kRows];
#pragma unroll
    for (int i = 0; i < kRows; ++i) {
        ap[i] = a + (row + i) * n;
    }

    for (int k = 0; k < n; k += 4) {
        float4 ar[kRows];
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            ar[i] = __ldg(reinterpret_cast<const float4*>(ap[i] + k));
        }
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(b + (k + 0) * n + col));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(b + (k + 1) * n + col));
        const float4 b2 = __ldg(reinterpret_cast<const float4*>(b + (k + 2) * n + col));
        const float4 b3 = __ldg(reinterpret_cast<const float4*>(b + (k + 3) * n + col));
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            acc[i][0] = fmaf(ar[i].x, b0.x, acc[i][0]);
            acc[i][1] = fmaf(ar[i].x, b0.y, acc[i][1]);
            acc[i][2] = fmaf(ar[i].x, b0.z, acc[i][2]);
            acc[i][3] = fmaf(ar[i].x, b0.w, acc[i][3]);
            acc[i][0] = fmaf(ar[i].y, b1.x, acc[i][0]);
            acc[i][1] = fmaf(ar[i].y, b1.y, acc[i][1]);
            acc[i][2] = fmaf(ar[i].y, b1.z, acc[i][2]);
            acc[i][3] = fmaf(ar[i].y, b1.w, acc[i][3]);
            acc[i][0] = fmaf(ar[i].z, b2.x, acc[i][0]);
            acc[i][1] = fmaf(ar[i].z, b2.y, acc[i][1]);
            acc[i][2] = fmaf(ar[i].z, b2.z, acc[i][2]);
            acc[i][3] = fmaf(ar[i].z, b2.w, acc[i][3]);
            acc[i][0] = fmaf(ar[i].w, b3.x, acc[i][0]);
            acc[i][1] = fmaf(ar[i].w, b3.y, acc[i][1]);
            acc[i][2] = fmaf(ar[i].w, b3.z, acc[i][2]);
            acc[i][3] = fmaf(ar[i].w, b3.w, acc[i][3]);
        }
    }

#pragma unroll
    for (int i = 0; i < kRows; ++i) {
        *reinterpret_cast<float4*>(c + (row + i) * n + col) =
            make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
    }
}

__global__ void gemm_tiny(const float* __restrict__ a, const float* __restrict__ b,
                          float* __restrict__ c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row * n + k], b[k * n + col], sum);
    }
    c[row * n + col] = sum;
}

struct GemmScratch {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    const float* hooked_a = nullptr;
    const float* hooked_b = nullptr;
    float* hooked_c = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t cap = 0;
};

GemmScratch& scratch() {
    static GemmScratch s;
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

void hook_const(const float*& slot, const float* ptr, std::size_t bytes) {
    if (slot == ptr) {
        return;
    }
    unhook(slot);
    if (cudaHostRegister(const_cast<float*>(ptr), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        slot = ptr;
    }
}

void hook_mut(float*& slot, float* ptr, std::size_t bytes) {
    if (slot == ptr) {
        return;
    }
    unhook_mut(slot);
    if (cudaHostRegister(ptr, bytes, cudaHostRegisterDefault) == cudaSuccess) {
        slot = ptr;
    }
}

void ensure_cap(std::size_t n) {
    GemmScratch& s = scratch();
    if (s.stream == nullptr) {
        cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
    }
    const std::size_t need = n * n;
    if (s.cap >= need) {
        return;
    }
    if (s.d_a != nullptr) {
        cudaFree(s.d_a);
        cudaFree(s.d_b);
        cudaFree(s.d_c);
    }
    const std::size_t bytes = need * sizeof(float);
    cudaMalloc(&s.d_a, bytes);
    cudaMalloc(&s.d_b, bytes);
    cudaMalloc(&s.d_c, bytes);
    s.cap = need;
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {
        return {};
    }

    ensure_cap(static_cast<std::size_t>(n));
    GemmScratch& s = scratch();
    const std::size_t bytes =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);
    hook_const(s.hooked_a, a.data(), bytes);
    hook_const(s.hooked_b, b.data(), bytes);

    cudaMemcpyAsync(s.d_a, a.data(), bytes, cudaMemcpyHostToDevice, s.stream);
    cudaMemcpyAsync(s.d_b, b.data(), bytes, cudaMemcpyHostToDevice, s.stream);

    std::vector<float> result(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

    const int tile_m = kTy * kRows;
    const int tile_n = kTx * kCols;
    if (n >= 128 && (n % tile_m) == 0 && (n % tile_n) == 0) {
        dim3 block(kTx, kTy);
        dim3 grid(n / tile_n, n / tile_m);
        gemm_coalesced<<<grid, block, 0, s.stream>>>(s.d_a, s.d_b, s.d_c, n);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        gemm_tiny<<<grid, block, 0, s.stream>>>(s.d_a, s.d_b, s.d_c, n);
    }

    hook_mut(s.hooked_c, result.data(), bytes);
    cudaMemcpyAsync(result.data(), s.d_c, bytes, cudaMemcpyDeviceToHost, s.stream);
    cudaStreamSynchronize(s.stream);
    return result;
}
