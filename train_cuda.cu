// GPU trainer. Runs the same architecture as model.cpp on CUDA, then writes a
// checkpoint through the CPU Model's own save path - so the file it produces is
// byte-identical in format to a CPU run, and ./infer loads it unchanged.
//
//   ./train_cuda <dataset.txt> <model.bin> [options]          pretrain on text
//   ./train_cuda <data.json> <tuned.bin> --init pre.bin       instruction tune
//
// Inference stays on the CPU by design; nothing here is needed to run a model.

#include "dataset.h"
#include "model.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t status = (call);                                                 \
        if (status != cudaSuccess) {                                                 \
            throw std::runtime_error(std::string("CUDA error at " __FILE__ ":") +    \
                                     std::to_string(__LINE__) + " - " +              \
                                     cudaGetErrorString(status));                    \
        }                                                                            \
    } while (0)

#define CUBLAS_CHECK(call)                                                           \
    do {                                                                             \
        cublasStatus_t status = (call);                                              \
        if (status != CUBLAS_STATUS_SUCCESS) {                                       \
            throw std::runtime_error(std::string("cuBLAS error at " __FILE__ ":") +  \
                                     std::to_string(__LINE__));                      \
        }                                                                            \
    } while (0)

namespace {

const int BLOCK = 256;

// ---------------------------------------------------------------------------
// Row-major GEMM on top of column-major cuBLAS.
//
// A row-major r x c buffer is bit-identical to a column-major c x r one, so
// asking cuBLAS for C^T = op(B)^T * op(A)^T with the operands swapped leaves
// exactly the row-major C we want in memory.
// ---------------------------------------------------------------------------
void gemm(cublasHandle_t handle, int m, int n, int k, const float* A, bool transposeA,
          const float* B, bool transposeB, float* C, float alpha = 1.0f, float beta = 0.0f) {
    // A contributes an m x k operand, B a k x n one, C is m x n - all row-major
    int lda = transposeA ? m : k;
    int ldb = transposeB ? k : n;
    cublasOperation_t opA = transposeA ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = transposeB ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(handle, opB, opA, n, m, k, &alpha, B, ldb, A, lda, &beta, C, n));
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void fill_kernel(float* values, size_t count, float value) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) values[i] = value;
}

__global__ void scale_kernel(float* values, size_t count, float factor) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) values[i] *= factor;
}

__global__ void add_kernel(float* dst, const float* src, size_t count) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) dst[i] += src[i];
}

__global__ void sgd_kernel(float* weights, const float* grads, size_t count, float lr) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < count) weights[i] -= lr * grads[i];
}

__global__ void sumsq_kernel(const float* values, size_t count, float* total) {
    __shared__ float shared[BLOCK];
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float value = (i < count) ? values[i] : 0.0f;
    shared[threadIdx.x] = value * value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(total, shared[0]);
}

// X[t] = table[ids[t]] + positional encoding, matching Embedding on the CPU
__global__ void embed_forward_kernel(const int* ids, const float* table, float* X, int T, int D) {
    int t = blockIdx.x;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float exponent = (float)(2 * (j / 2)) / (float)D;
        float angle = (float)t / powf(10000.0f, exponent);
        float positional = (j % 2 == 0) ? sinf(angle) : cosf(angle);
        X[(size_t)t * D + j] = table[(size_t)ids[t] * D + j] + positional;
    }
}

// Scatter each row of dX back onto the token that produced it
__global__ void embed_backward_kernel(const int* ids, const float* dX, float* dtable,
                                      int T, int D) {
    int t = blockIdx.x;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        atomicAdd(&dtable[(size_t)ids[t] * D + j], dX[(size_t)t * D + j]);
    }
}

// Row-wise softmax over the causal window [0, i]. Masking is implicit: the loop
// simply never looks past the diagonal, and the tail is written as zero.
__global__ void softmax_causal_kernel(float* S, int T) {
    __shared__ float shared[BLOCK];
    int i = blockIdx.x;
    float* row = S + (size_t)i * T;
    int width = i + 1;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        local_max = fmaxf(local_max, row[j]);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_value = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        float e = __expf(row[j] - max_value);
        row[j] = e;
        local_sum += e;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / shared[0];
    __syncthreads();

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        row[j] *= inv_sum;
    }
    for (int j = width + threadIdx.x; j < T; j += blockDim.x) {
        row[j] = 0.0f; // Future positions contribute nothing
    }
}

// dS = A * (dA - sum(dA * A)), over the same causal window
__global__ void softmax_causal_backward_kernel(const float* A, const float* dA, float* dS, int T) {
    __shared__ float shared[BLOCK];
    int i = blockIdx.x;
    const float* a_row = A + (size_t)i * T;
    const float* da_row = dA + (size_t)i * T;
    float* ds_row = dS + (size_t)i * T;
    int width = i + 1;

    float local_dot = 0.0f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        local_dot += da_row[j] * a_row[j];
    }
    shared[threadIdx.x] = local_dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float dot = shared[0];
    __syncthreads();

    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        ds_row[j] = a_row[j] * (da_row[j] - dot);
    }
    for (int j = width + threadIdx.x; j < T; j += blockDim.x) {
        ds_row[j] = 0.0f;
    }
}

// Y = gamma * (x - mean) / sqrt(var + eps) + beta, one block per token
__global__ void layernorm_forward_kernel(const float* X, const float* gamma, const float* beta,
                                         float* Y, float* normalized, float* inv_std,
                                         int T, int D) {
    __shared__ float shared[BLOCK];
    int t = blockIdx.x;
    const float* x = X + (size_t)t * D;
    float* y = Y + (size_t)t * D;
    float* norm = normalized + (size_t)t * D;

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) local_sum += x[j];
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float mean = shared[0] / (float)D;
    __syncthreads();

    float local_var = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float diff = x[j] - mean;
        local_var += diff * diff;
    }
    shared[threadIdx.x] = local_var;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float variance = shared[0] / (float)D;
    float scale = rsqrtf(variance + 1e-5f);
    if (threadIdx.x == 0) inv_std[t] = scale;
    __syncthreads();

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float n = (x[j] - mean) * scale;
        norm[j] = n;
        y[j] = gamma[j] * n + beta[j];
    }
}

// dx = inv_std * (dnorm - mean(dnorm) - normalized * mean(dnorm * normalized))
__global__ void layernorm_backward_kernel(const float* dY, const float* normalized,
                                          const float* inv_std, const float* gamma,
                                          float* dX, float* dgamma, float* dbeta,
                                          int T, int D) {
    __shared__ float shared_a[BLOCK];
    __shared__ float shared_b[BLOCK];
    int t = blockIdx.x;
    const float* dy = dY + (size_t)t * D;
    const float* norm = normalized + (size_t)t * D;
    float* dx = dX + (size_t)t * D;

    float local_dnorm = 0.0f;
    float local_dnorm_norm = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float dnorm = dy[j] * gamma[j];
        local_dnorm += dnorm;
        local_dnorm_norm += dnorm * norm[j];
        // Scale and shift are per-feature, so their gradients sum over tokens
        atomicAdd(&dgamma[j], dy[j] * norm[j]);
        atomicAdd(&dbeta[j], dy[j]);
    }
    shared_a[threadIdx.x] = local_dnorm;
    shared_b[threadIdx.x] = local_dnorm_norm;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_a[threadIdx.x] += shared_a[threadIdx.x + stride];
            shared_b[threadIdx.x] += shared_b[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float mean_dnorm = shared_a[0] / (float)D;
    float mean_dnorm_norm = shared_b[0] / (float)D;
    float scale = inv_std[t];
    __syncthreads();

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float dnorm = dy[j] * gamma[j];
        dx[j] = scale * (dnorm - mean_dnorm - norm[j] * mean_dnorm_norm);
    }
}

// Add the output bias, then softmax each row over the vocabulary
__global__ void softmax_logits_kernel(float* logits, const float* bias, int T, int V) {
    __shared__ float shared[BLOCK];
    int t = blockIdx.x;
    float* row = logits + (size_t)t * V;

    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        row[j] += bias[j];
        local_max = fmaxf(local_max, row[j]);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_value = shared[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        float e = __expf(row[j] - max_value);
        row[j] = e;
        local_sum += e;
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    float inv_sum = 1.0f / shared[0];
    __syncthreads();

    for (int j = threadIdx.x; j < V; j += blockDim.x) {
        row[j] *= inv_sum;
    }
}

// Mean cross-entropy over the graded positions only
__global__ void cross_entropy_kernel(const float* probs, const int* targets, const char* mask,
                                     float* loss_out, int T, int V) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T || !mask[t]) return;
    float p = probs[(size_t)t * V + targets[t]];
    atomicAdd(loss_out, -__logf(p + 1e-9f));
}

// (probs - one_hot) / graded_count, with masked rows left at zero
__global__ void dlogits_kernel(float* dLogits, const float* probs, const int* targets,
                               const char* mask, float inv_counted, int T, int V) {
    int t = blockIdx.y;
    if (!mask[t]) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < V;
             j += gridDim.x * blockDim.x) {
            dLogits[(size_t)t * V + j] = 0.0f;
        }
        return;
    }
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < V; j += gridDim.x * blockDim.x) {
        float value = probs[(size_t)t * V + j] * inv_counted;
        if (j == targets[t]) value -= inv_counted;
        dLogits[(size_t)t * V + j] = value;
    }
}

// db = column sums of dLogits
__global__ void colsum_kernel(const float* dLogits, float* db, int T, int V) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= V) return;
    float total = 0.0f;
    for (int t = 0; t < T; ++t) {
        total += dLogits[(size_t)t * V + j];
    }
    db[j] += total;
}

int grid_for(size_t count) {
    return (int)((count + BLOCK - 1) / BLOCK);
}

// ---------------------------------------------------------------------------
// Device-side model: weights, gradients, and the activation workspace
// ---------------------------------------------------------------------------

struct GpuBuffer {
    float* data = nullptr;
    size_t count = 0;

    void allocate(size_t elements) {
        count = elements;
        CUDA_CHECK(cudaMalloc(&data, elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(data, 0, elements * sizeof(float)));
    }
    void free() {
        if (data) cudaFree(data);
        data = nullptr;
    }
};

class GpuTrainer {
public:
    GpuTrainer(Model& cpu_model, int max_seq)
        : model(cpu_model),
          V(cpu_model.config.vocab_size),
          D(cpu_model.config.embed_dim),
          loops(cpu_model.config.num_loops),
          max_T(max_seq) {
        if (cpu_model.config.head_dim != D) {
            throw std::runtime_error("the CUDA trainer needs head_dim == embed_dim");
        }
        CUBLAS_CHECK(cublasCreate(&blas));

        // Weights and their gradients, in the order Model::parameters() reports
        params = model.parameters();
        weights.resize(params.size());
        grads.resize(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            weights[i].allocate(params[i].size());
            grads[i].allocate(params[i].size());
        }

        size_t TD = (size_t)max_T * D;
        size_t TT = (size_t)max_T * max_T;
        size_t TV = (size_t)max_T * V;

        X.resize(loops + 1);
        for (auto& buffer : X) buffer.allocate(TD);
        Q.resize(loops); K.resize(loops); Vb.resize(loops);
        A.resize(loops); normalized.resize(loops); inv_std.resize(loops);
        for (int l = 0; l < loops; ++l) {
            Q[l].allocate(TD); K[l].allocate(TD); Vb[l].allocate(TD);
            A[l].allocate(TT);
            normalized[l].allocate(TD);
            inv_std[l].allocate(max_T);
        }

        attn_out.allocate(TD);
        resid.allocate(TD);
        logits.allocate(TV);
        dLogits.allocate(TV);
        dX.allocate(TD);
        dY.allocate(TD);
        dAttn.allocate(TD);
        dA.allocate(TT);
        dS.allocate(TT);
        dQ.allocate(TD); dK.allocate(TD); dV.allocate(TD);

        CUDA_CHECK(cudaMalloc(&ids, (size_t)max_T * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&targets, (size_t)max_T * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&mask, (size_t)max_T * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&scalar, 2 * sizeof(float)));

        upload_weights();
    }

    ~GpuTrainer() {
        for (auto& buffer : weights) buffer.free();
        for (auto& buffer : grads) buffer.free();
        for (auto& buffer : X) buffer.free();
        for (auto& buffer : Q) buffer.free();
        for (auto& buffer : K) buffer.free();
        for (auto& buffer : Vb) buffer.free();
        for (auto& buffer : A) buffer.free();
        for (auto& buffer : normalized) buffer.free();
        for (auto& buffer : inv_std) buffer.free();
        attn_out.free(); resid.free(); logits.free(); dLogits.free();
        dX.free(); dY.free(); dAttn.free(); dA.free(); dS.free();
        dQ.free(); dK.free(); dV.free();
        if (ids) cudaFree(ids);
        if (targets) cudaFree(targets);
        if (mask) cudaFree(mask);
        if (scalar) cudaFree(scalar);
        cublasDestroy(blas);
    }

    void upload_weights() {
        for (size_t i = 0; i < params.size(); ++i) {
            CUDA_CHECK(cudaMemcpy(weights[i].data, params[i].data,
                                  params[i].size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void download_weights() {
        for (size_t i = 0; i < params.size(); ++i) {
            CUDA_CHECK(cudaMemcpy(params[i].data, weights[i].data,
                                  params[i].size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // One masked SGD step on one sequence; returns the mean loss over graded tokens
    float train_step(const Example& example, float learning_rate, float max_grad_norm) {
        int T = (int)example.tokens.size() - 1; // Inputs; targets are shifted by one
        if (T < 1) return 0.0f;
        if (T > max_T) throw std::runtime_error("sequence longer than the allocated workspace");

        // The mask shifts with the targets: position t predicts tokens[t + 1]
        std::vector<char> target_mask(example.train_on.begin() + 1, example.train_on.end());
        int counted = 0;
        for (char flag : target_mask) {
            if (flag) ++counted;
        }
        if (counted == 0) return 0.0f;

        CUDA_CHECK(cudaMemcpy(ids, example.tokens.data(), T * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(targets, example.tokens.data() + 1, T * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mask, target_mask.data(), T * sizeof(char),
                              cudaMemcpyHostToDevice));

        zero_grads();
        forward(T);
        float loss = compute_loss(T, counted);
        backward(T, counted);
        clip_and_update(learning_rate, max_grad_norm);
        return loss;
    }

private:
    Model& model;
    int V, D, loops, max_T;
    cublasHandle_t blas;

    std::vector<ParamRef> params;
    std::vector<GpuBuffer> weights, grads;

    // Parameter indices, matching Model::parameters() order
    enum { P_TABLE = 0, P_WQ, P_WK, P_WV, P_GAMMA, P_BETA, P_WO, P_BO };

    std::vector<GpuBuffer> X, Q, K, Vb, A, normalized, inv_std;
    GpuBuffer attn_out, resid, logits, dLogits, dX, dY, dAttn, dA, dS, dQ, dK, dV;
    int* ids = nullptr;
    int* targets = nullptr;
    char* mask = nullptr;
    float* scalar = nullptr;

    void zero_grads() {
        for (auto& buffer : grads) {
            CUDA_CHECK(cudaMemset(buffer.data, 0, buffer.count * sizeof(float)));
        }
    }

    void forward(int T) {
        // Embed once, positions included
        embed_forward_kernel<<<T, BLOCK>>>(ids, weights[P_TABLE].data, X[0].data, T, D);

        float scale = 1.0f / std::sqrt((float)D);
        for (int l = 0; l < loops; ++l) {
            // Q, K, V = X * W
            gemm(blas, T, D, D, X[l].data, false, weights[P_WQ].data, false, Q[l].data);
            gemm(blas, T, D, D, X[l].data, false, weights[P_WK].data, false, K[l].data);
            gemm(blas, T, D, D, X[l].data, false, weights[P_WV].data, false, Vb[l].data);

            // Scores = Q * K^T / sqrt(D), then a causal softmax
            gemm(blas, T, T, D, Q[l].data, false, K[l].data, true, A[l].data, scale);
            softmax_causal_kernel<<<T, BLOCK>>>(A[l].data, T);

            // Attention output, then Norm(x + g(x))
            gemm(blas, T, D, T, A[l].data, false, Vb[l].data, false, attn_out.data);
            CUDA_CHECK(cudaMemcpy(resid.data, attn_out.data, (size_t)T * D * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
            add_kernel<<<grid_for((size_t)T * D), BLOCK>>>(resid.data, X[l].data, (size_t)T * D);
            layernorm_forward_kernel<<<T, BLOCK>>>(resid.data, weights[P_GAMMA].data,
                                                   weights[P_BETA].data, X[l + 1].data,
                                                   normalized[l].data, inv_std[l].data, T, D);
        }

        // Only the final iteration reaches the head
        gemm(blas, T, V, D, X[loops].data, false, weights[P_WO].data, false, logits.data);
        softmax_logits_kernel<<<T, BLOCK>>>(logits.data, weights[P_BO].data, T, V);
    }

    float compute_loss(int T, int counted) {
        CUDA_CHECK(cudaMemset(scalar, 0, sizeof(float)));
        cross_entropy_kernel<<<grid_for(T), BLOCK>>>(logits.data, targets, mask, scalar, T, V);

        float total = 0.0f;
        CUDA_CHECK(cudaMemcpy(&total, scalar, sizeof(float), cudaMemcpyDeviceToHost));
        return total / (float)counted;
    }

    void backward(int T, int counted) {
        float inv_counted = 1.0f / (float)counted;
        dim3 logits_grid(std::min(64, (V + BLOCK - 1) / BLOCK), T);
        dlogits_kernel<<<logits_grid, BLOCK>>>(dLogits.data, logits.data, targets, mask,
                                               inv_counted, T, V);

        // Head: dW = X^T * dLogits, db = column sums, dX = dLogits * W^T
        gemm(blas, D, V, T, X[loops].data, true, dLogits.data, false, grads[P_WO].data,
             1.0f, 1.0f);
        colsum_kernel<<<grid_for(V), BLOCK>>>(dLogits.data, grads[P_BO].data, T, V);
        gemm(blas, T, D, V, dLogits.data, false, weights[P_WO].data, true, dY.data);

        float scale = 1.0f / std::sqrt((float)D);
        size_t TD = (size_t)T * D;

        for (int l = loops - 1; l >= 0; --l) {
            // Through the norm
            layernorm_backward_kernel<<<T, BLOCK>>>(dY.data, normalized[l].data,
                                                    inv_std[l].data, weights[P_GAMMA].data,
                                                    dAttn.data, grads[P_GAMMA].data,
                                                    grads[P_BETA].data, T, D);

            // The residual splits: dAttn feeds the block, and also flows around it
            gemm(blas, T, T, D, dAttn.data, false, Vb[l].data, true, dA.data);
            gemm(blas, T, D, T, A[l].data, true, dAttn.data, false, dV.data);
            softmax_causal_backward_kernel<<<T, BLOCK>>>(A[l].data, dA.data, dS.data, T);

            gemm(blas, T, D, T, dS.data, false, K[l].data, false, dQ.data, scale);
            gemm(blas, T, D, T, dS.data, true, Q[l].data, false, dK.data, scale);

            // Shared weights, so these accumulate across every iteration
            gemm(blas, D, D, T, X[l].data, true, dQ.data, false, grads[P_WQ].data, 1.0f, 1.0f);
            gemm(blas, D, D, T, X[l].data, true, dK.data, false, grads[P_WK].data, 1.0f, 1.0f);
            gemm(blas, D, D, T, X[l].data, true, dV.data, false, grads[P_WV].data, 1.0f, 1.0f);

            gemm(blas, T, D, D, dQ.data, false, weights[P_WQ].data, true, dX.data);
            gemm(blas, T, D, D, dK.data, false, weights[P_WK].data, true, dX.data, 1.0f, 1.0f);
            gemm(blas, T, D, D, dV.data, false, weights[P_WV].data, true, dX.data, 1.0f, 1.0f);

            add_kernel<<<grid_for(TD), BLOCK>>>(dX.data, dAttn.data, TD); // Skip path
            CUDA_CHECK(cudaMemcpy(dY.data, dX.data, TD * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }

        embed_backward_kernel<<<T, BLOCK>>>(ids, dY.data, grads[P_TABLE].data, T, D);
    }

    void clip_and_update(float learning_rate, float max_grad_norm) {
        CUDA_CHECK(cudaMemset(scalar, 0, sizeof(float)));
        for (auto& buffer : grads) {
            sumsq_kernel<<<grid_for(buffer.count), BLOCK>>>(buffer.data, buffer.count, scalar);
        }

        float sum_sq = 0.0f;
        CUDA_CHECK(cudaMemcpy(&sum_sq, scalar, sizeof(float), cudaMemcpyDeviceToHost));
        float total = std::sqrt(sum_sq);

        float factor = 1.0f;
        if (!std::isfinite(total)) {
            factor = 0.0f; // An overflowed step would poison every weight; drop it
        } else if (max_grad_norm > 0.0f && total > max_grad_norm) {
            factor = max_grad_norm / (total + 1e-6f);
        }

        for (size_t i = 0; i < grads.size(); ++i) {
            if (factor != 1.0f) {
                scale_kernel<<<grid_for(grads[i].count), BLOCK>>>(grads[i].data,
                                                                  grads[i].count, factor);
            }
            sgd_kernel<<<grid_for(weights[i].count), BLOCK>>>(weights[i].data, grads[i].data,
                                                              weights[i].count, learning_rate);
        }
    }
};

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct Options {
    std::string data_path;
    std::string output_path;
    std::string init_path; // Empty means pretrain from scratch
    int epochs = 300;
    int merges = 200;
    int dim = 32;
    int loops = 4;
    float learning_rate = 0.01f;
    float max_grad_norm = 1.0f;
    int max_tokens = 8132;
    unsigned seed = 1234;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <dataset> <out.bin> [options]\n"
              << "\n"
              << "  <dataset> is a .txt of one string per line, or a .json of\n"
              << "  conversations when --init names a pretrained checkpoint.\n"
              << "\n"
              << "Options:\n"
              << "  --init F       tune this checkpoint instead of training from scratch\n"
              << "  --epochs N     passes over the dataset (default 300)\n"
              << "  --merges N     BPE merges, pretraining only (default 200)\n"
              << "  --dim N        width, pretraining only (default 32)\n"
              << "  --loops N      attention iterations, pretraining only (default 4)\n"
              << "  --lr F         SGD learning rate (default 0.01)\n"
              << "  --clip F       max global gradient norm, 0 disables (default 1.0)\n"
              << "  --max-tokens N longest sequence to train on (default 8132)\n"
              << "  --seed N       shuffle seed (default 1234)\n";
}

bool parse_options(int argc, char** argv, Options& opts) {
    if (argc < 3) return false;
    opts.data_path = argv[1];
    opts.output_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << flag << "\n";
            return false;
        }
        std::string value = argv[++i];

        if (flag == "--init") opts.init_path = value;
        else if (flag == "--epochs") opts.epochs = std::stoi(value);
        else if (flag == "--merges") opts.merges = std::stoi(value);
        else if (flag == "--dim") opts.dim = std::stoi(value);
        else if (flag == "--loops") opts.loops = std::stoi(value);
        else if (flag == "--lr") opts.learning_rate = std::stof(value);
        else if (flag == "--clip") opts.max_grad_norm = std::stof(value);
        else if (flag == "--max-tokens") opts.max_tokens = std::stoi(value);
        else if (flag == "--seed") opts.seed = static_cast<unsigned>(std::stoul(value));
        else {
            std::cerr << "Unknown option: " << flag << "\n";
            return false;
        }
    }
    return true;
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void report_device() {
    int device = 0;
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    std::cout << "GPU: " << properties.name << " (sm_" << properties.major << properties.minor
              << "), " << (free_bytes >> 20) << " MB free of " << (total_bytes >> 20) << " MB"
              << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_options(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        report_device();

        bool tuning = !opts.init_path.empty();
        std::vector<Example> raw;

        // The tokenizer decides how the data encodes, so the model comes first.
        // Tuning loads one; pretraining builds it around a freshly learned vocabulary.
        std::optional<Model> holder;

        if (tuning) {
            holder = Model::load(opts.init_path);
            Model& model = *holder;
            std::cout << "Loaded " << opts.init_path << ": vocab " << model.config.vocab_size
                      << ", dim " << model.config.embed_dim << ", " << model.config.num_loops
                      << " attention loops" << std::endl;

            if (ends_with(opts.data_path, ".json")) {
                raw = build_chat_examples(model.tokenizer, json_parse_file(opts.data_path));
            } else {
                raw = build_text_examples(model.tokenizer, read_text_lines(opts.data_path));
            }
        } else {
            // Pretraining: learn the vocabulary, then size the model around it
            std::vector<std::string> lines = read_text_lines(opts.data_path);
            if (lines.empty()) {
                std::cerr << "Dataset '" << opts.data_path << "' has no usable lines\n";
                return 1;
            }
            std::cout << "Dataset: " << lines.size() << " lines from " << opts.data_path
                      << std::endl;

            BPETokenizer tokenizer;
            tokenizer.train(lines, opts.merges);
            std::cout << "Vocabulary: " << tokenizer.vocab_size() << " tokens from "
                      << tokenizer.num_merges() << " merges" << std::endl;

            ModelConfig config;
            config.embed_dim = opts.dim;
            config.head_dim = opts.dim;
            config.vocab_size = tokenizer.vocab_size();
            config.num_loops = opts.loops;

            raw = build_text_examples(tokenizer, lines);
            holder.emplace(config, std::move(tokenizer));
        }

        Model& model = *holder;

        DatasetStats stats;
        std::vector<Example> examples =
            prepare_examples(model.tokenizer, std::move(raw), opts.max_tokens, stats);
        if (examples.empty()) {
            std::cerr << "No sequence had anything to train on\n";
            return 1;
        }
        print_dataset_stats(stats, opts.max_tokens);

        // The workspace is sized to the longest sequence that actually survives
        // truncation, not to --max-tokens, so a short dataset costs little
        size_t longest = 0;
        for (const auto& example : examples) {
            longest = std::max(longest, example.tokens.size());
        }
        int max_T = (int)longest;

        // The T x T attention matrices are what fill the card: one per loop
        // iteration, plus dA and dS during the backward pass
        size_t attention_bytes = (size_t)max_T * max_T * sizeof(float) *
                                 (size_t)(model.config.num_loops + 2);
        std::cout << "Workspace: " << max_T << " tokens, " << (attention_bytes >> 20)
                  << " MB of attention matrices" << std::endl;

        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (attention_bytes > free_bytes) {
            // Solve (loops + 2) * T^2 * 4 <= free for the length that would fit
            double budget = (double)free_bytes /
                            (4.0 * (double)(model.config.num_loops + 2));
            std::cerr << "Error: the longest sequence needs " << (attention_bytes >> 20)
                      << " MB but only " << (free_bytes >> 20) << " MB is free.\n"
                      << "Attention memory grows with the square of the sequence: at "
                      << model.config.num_loops << " loops this GPU tops out near "
                      << (long)std::sqrt(budget) << " tokens.\n"
                      << "Lower --max-tokens, or --loops, to fit.\n";
            return 1;
        }

        GpuTrainer trainer(model, max_T);

        std::cout << "\nTraining " << opts.epochs << " epochs at lr " << opts.learning_rate
                  << ", " << model.config.num_loops << " attention loops, clip "
                  << opts.max_grad_norm << ":" << std::endl;

        std::vector<size_t> order(examples.size());
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 gen(opts.seed);
        int report_every = std::max(1, opts.epochs / 20);

        for (int epoch = 0; epoch < opts.epochs; ++epoch) {
            std::shuffle(order.begin(), order.end(), gen);

            float epoch_loss = 0.0f;
            for (size_t index : order) {
                epoch_loss += trainer.train_step(examples[index], opts.learning_rate,
                                                 opts.max_grad_norm);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            if (epoch % report_every == 0 || epoch == opts.epochs - 1) {
                std::cout << "  epoch " << epoch << "\tloss "
                          << epoch_loss / (float)examples.size() << std::endl;
            }
        }

        // Weights come home, and the CPU model writes the checkpoint
        trainer.download_weights();
        model.save(opts.output_path);
        std::cout << "\nSaved checkpoint to " << opts.output_path << std::endl;
        std::cout << "Run it on the CPU with: ./infer " << opts.output_path
                  << " \"your prompt\"" << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }

    return 0;
}
