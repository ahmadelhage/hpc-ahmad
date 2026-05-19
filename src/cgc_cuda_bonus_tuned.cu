// ========================= FIXES APPLIED =========================
//
// 1. Re-enabled REAL autotuning between ATOMIC and SHARED
// 2. REMOVED broken warp kernel completely
// 3. Added NaN-safe distance accumulation
// 4. Removed unnecessary device-wide synchronizations
// 5. Fixed shared-memory alignment
// 6. Fixed autotune pollution completely
//
// ================================================================

#include <chrono>
#include <iostream>
#include <limits>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <functional>

#include "common.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
        }                                                                       \
    } while (0)

// ================================================================
// KERNEL 1 — GLOBAL ATOMICS
// ================================================================

__global__ void kernel_cluster_sum(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels,
    double* local_sum,
    int* local_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_rows * local_cols)
        return;

    int row = idx / local_cols;
    int col = idx % local_cols;

    double item = (double)matrix[row * local_cols + col];

    int cluster =
        row_labels[row] * num_col_labels + col_labels[col];

    atomicAdd(&local_sum[cluster], item);
    atomicAdd(&local_count[cluster], 1);
}

// ================================================================
// KERNEL 1 — SHARED MEMORY VERSION
// ================================================================

__global__ void kernel_cluster_sum_shared(
    int num_rows,
    int local_cols,
    int num_col_labels,
    int num_clusters,
    const float* matrix,
    const label_type* row_labels,
    const label_type* col_labels,
    double* global_sum,
    int* global_count)
{
    extern __shared__ double s_sum[];

    int* s_count = (int*)&s_sum[num_clusters];

    for (int i = threadIdx.x;
         i < num_clusters;
         i += blockDim.x)
    {
        s_sum[i] = 0.0;
        s_count[i] = 0;
    }

    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_rows * local_cols) {

        int row = idx / local_cols;
        int col = idx % local_cols;

        double item =
            (double)matrix[row * local_cols + col];

        int cluster =
            row_labels[row] * num_col_labels + col_labels[col];

        atomicAdd(&s_sum[cluster], item);
        atomicAdd(&s_count[cluster], 1);
    }

    __syncthreads();

    for (int i = threadIdx.x;
         i < num_clusters;
         i += blockDim.x)
    {
        if (s_count[i] > 0) {
            atomicAdd(&global_sum[i], s_sum[i]);
            atomicAdd(&global_count[i], s_count[i]);
        }
    }
}

// ================================================================
// KERNEL 2 — DIVIDE AVG
// ================================================================

__global__ void kernel_divide_avg(
    int num_clusters,
    const double* global_sum,
    const int* global_count,
    double* cluster_avg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_clusters)
        return;

    cluster_avg[idx] =
        global_sum[idx] / (double)global_count[idx];
}

// ================================================================
// KERNEL 3 — ROW DISTANCES
// ================================================================

__global__ void kernel_row_distances(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    const label_type* col_labels,
    const double* cluster_avg,
    double* partial_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_rows * num_row_labels)
        return;

    int row = idx / num_row_labels;
    int k   = idx % num_row_labels;

    double dist = 0.0;

    for (int col = 0; col < local_cols; col++) {

        double item =
            (double)matrix[row * local_cols + col];

        int cl =
            k * num_col_labels + col_labels[col];

        double avg = cluster_avg[cl];

        // ===== FIX =====
        if (!isnan(avg)) {
            double diff = avg - item;
            dist += diff * diff;
        }
    }

    partial_dist[row * num_row_labels + k] = dist;
}

// ================================================================
// KERNEL 4 — COLUMN LABELS
// ================================================================

__global__ void kernel_col_labels(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* matrix,
    const label_type* row_labels,
    label_type* col_labels,
    const double* cluster_avg,
    int* num_updated)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col >= local_cols)
        return;

    int best_label = 0;
    double best_dist = 1e300;

    for (int k = 0; k < num_col_labels; k++) {

        double dist = 0.0;

        for (int row = 0; row < num_rows; row++) {

            double item =
                (double)matrix[row * local_cols + col];

            int cl =
                row_labels[row] * num_col_labels + k;

            double avg = cluster_avg[cl];

            // ===== FIX =====
            if (!isnan(avg)) {
                double diff = avg - item;
                dist += diff * diff;
            }
        }

        // ===== POSSIBLE TIEBREAK FIX =====
        if (dist <= best_dist) {
            best_dist = dist;
            best_label = k;
        }
    }

    if (col_labels[col] != best_label) {
        col_labels[col] = best_label;
        atomicAdd(num_updated, 1);
    }
}

enum class SumVariant {
    ATOMIC,
    SHARED
};

struct TunedConfig {
    int cluster_sum;
    SumVariant sum_variant;
    int divide_avg;
    int row_dist;
    int col_labels;
};

// ================================================================
// KERNEL TIMER
// ================================================================

static float time_kernel_ms(
    std::function<void()> fn,
    int warmup = 2,
    int runs = 5)
{
    for (int i = 0; i < warmup; i++) {
        fn();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < runs; i++) {
        fn();
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / runs;
}

// ================================================================
// AUTOTUNER
// ================================================================

TunedConfig auto_tune(
    int rank,
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    int num_clusters,
    float* d_matrix,
    label_type* d_row_labels,
    label_type* d_col_labels,
    double* d_cluster_avg,
    double* d_global_sum,
    int* d_global_count,
    double* d_partial_dist,
    int* d_cols_updated)
{
    const std::vector<int> candidates = {
        64, 128, 256, 512
    };

    TunedConfig best = {
        256,
        SumVariant::SHARED,
        256,
        256,
        256
    };

    size_t shared_size =
        num_clusters * (sizeof(double) + sizeof(int));

    // ============================================================
    // TUNE CLUSTER SUM
    // ============================================================

    {
        float best_ms = 1e9f;

        for (int t : candidates) {

            int blocks =
                (num_rows * local_cols + t - 1) / t;

            float ms_a = time_kernel_ms([&]() {

                CUDA_CHECK(cudaMemset(
                    d_global_sum,
                    0,
                    num_clusters * sizeof(double)));

                CUDA_CHECK(cudaMemset(
                    d_global_count,
                    0,
                    num_clusters * sizeof(int)));

                kernel_cluster_sum<<<blocks, t>>>(
                    num_rows,
                    local_cols,
                    num_col_labels,
                    d_matrix,
                    d_row_labels,
                    d_col_labels,
                    d_global_sum,
                    d_global_count);
            });

            float ms_b = time_kernel_ms([&]() {

                CUDA_CHECK(cudaMemset(
                    d_global_sum,
                    0,
                    num_clusters * sizeof(double)));

                CUDA_CHECK(cudaMemset(
                    d_global_count,
                    0,
                    num_clusters * sizeof(int)));

                kernel_cluster_sum_shared<<<
                    blocks,
                    t,
                    shared_size>>>(
                        num_rows,
                        local_cols,
                        num_col_labels,
                        num_clusters,
                        d_matrix,
                        d_row_labels,
                        d_col_labels,
                        d_global_sum,
                        d_global_count);
            });

            float best_at_t = ms_a;
            SumVariant var = SumVariant::ATOMIC;

            // ===== FIX =====
            if (ms_b < best_at_t) {
                best_at_t = ms_b;
                var = SumVariant::SHARED;
            }

            if (best_at_t < best_ms) {
                best_ms = best_at_t;
                best.cluster_sum = t;
                best.sum_variant = var;
            }
        }
    }

    return best;
}