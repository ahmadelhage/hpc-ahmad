#include <chrono>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <functional>

#include "common.h"

// CUDA error-check helper

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
        }                                                                       \
    } while (0)

// KERNEL 1 (benchmark): accumulate partial cluster sums and counts using global atomics
//   One thread per matrix entry (row, col).
//   Uses atomicAdd to accumulate into shared cluster arrays.
__global__ void kernel_cluster_sum(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float*      matrix,
    const label_type* row_labels,
    const label_type* col_labels,
    double*           local_sum,
    int*              local_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows * local_cols) return;

    int row = idx / local_cols;
    int col = idx % local_cols;

    double    item    = (double)matrix[row * local_cols + col];
    int       cluster = row_labels[row] * num_col_labels + col_labels[col];

    atomicAdd(&local_sum[cluster],   item);
    atomicAdd(&local_count[cluster], 1);
}

// KERNEL 1 bonus variant: use shared memory to reduce global atomics
// Each block accumulates into shared memory, then one thread per cluster does a global atomic add.
// This can be much faster when there are many threads updating the same cluster, as it reduces 
//contention on global atomics.
__global__ void kernel_cluster_sum_shared(
    int num_rows, int local_cols, int num_col_labels, int num_clusters,
    const float* matrix, const label_type* row_labels,
    const label_type* col_labels, double* global_sum, int* global_count)
{
    // Shared memory layout: first num_clusters doubles for sums, then num_clusters ints for counts
    extern __shared__ char smem[];
    double* s_sum   = reinterpret_cast<double*>(smem);
    int*    s_count = reinterpret_cast<int*>(s_sum + num_clusters);

    // Initialize shared memory accumulators to zero
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        s_sum[i]   = 0.0;
        s_count[i] = 0;
    }
    // Sync to make sure all threads see the initialized shared memory
    __syncthreads();

    // Each thread processes one matrix entry and updates shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows * local_cols) {
        int row     = idx / local_cols;
        int col     = idx % local_cols;
        double item = (double)matrix[row * local_cols + col];
        // Compute cluster index for this (row, col) pair
        int cluster = row_labels[row] * num_col_labels + col_labels[col];
        atomicAdd(&s_sum[cluster],   item);
        atomicAdd(&s_count[cluster], 1);
    }
    __syncthreads();

    // One global atomic per cluster per block
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        if (s_count[i] > 0) {
        atomicAdd(&global_sum[i],   s_sum[i]);
        atomicAdd(&global_count[i], s_count[i]);
        }
    }
}

// KERNEL 1 bonus variant 2: use warp-level primitives to reduce within warps before global atomics
// This can be even faster than shared memory when there are many threads updating the same cluster,
// as it avoids shared memory bank conflicts and synchronization. However, it only reduces within warps,
// so it may not be as effective if there are many threads per cluster that span multiple warps.
__global__ void kernel_cluster_sum_warp(
    int num_rows, int local_cols, int num_col_labels, int num_clusters,
    const float* matrix, const label_type* row_labels,
    const label_type* col_labels, double* global_sum, int* global_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows * local_cols) return;

    int row     = idx / local_cols;
    int col     = idx % local_cols;
    double item = (double)matrix[row * local_cols + col];
    int cluster = row_labels[row] * num_col_labels + col_labels[col];

    // Warp-level reduction: threads with matching cluster accumulate together
    // using __shfl_down_sync to pass values between lanes without shared memory
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;

    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_val     = __shfl_down_sync(mask, item,    offset);
        int    other_cluster = __shfl_down_sync(mask, cluster, offset);
        if (other_cluster == cluster)
            item += other_val;
    }

    // Only the lowest-lane thread of each matching group writes to global memory
    int prev_cluster = __shfl_up_sync(mask, cluster, 1);
    if (lane == 0 || prev_cluster != cluster) {
        atomicAdd(&global_sum[cluster],   item);
        atomicAdd(&global_count[cluster], 1);
    }
}


// KERNEL 2: divide global sums by counts to get cluster averages
//   One thread per cluster.
__global__ void kernel_divide_avg(
    int           num_clusters,
    const double* global_sum,
    const int*    global_count,
    double*       cluster_avg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clusters) return;

    cluster_avg[idx] = global_sum[idx] / (double)global_count[idx];
}

// KERNEL 3: compute partial row-label distances
//   One thread per (row, candidate_label) pair.
//   Each thread loops over its local columns to accumulate the partial distance.
__global__ void kernel_row_distances(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float*      matrix,
    const label_type* col_labels,
    const double*     cluster_avg,
    double*           partial_dist)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows * num_row_labels) return;

    int row = idx / num_row_labels;
    int k   = idx % num_row_labels;

    double dist = 0.0;
    for (int col = 0; col < local_cols; col++) {
        double item = (double)matrix[row * local_cols + col];
        int    cl   = k * num_col_labels + col_labels[col];
        double diff = cluster_avg[cl] - item;
        dist += diff * diff / (cluster_avg[cl] + 1e-9); // Add small epsilon to avoid division by zero
    }
    partial_dist[row * num_row_labels + k] = dist;
}

// KERNEL 4: update column labels
//   One thread per local column.
//   Each thread tries all candidate col-labels and picks the closest cluster.
__global__ void kernel_col_labels(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float*      matrix,
    const label_type* row_labels,
    label_type*       col_labels,
    const double*     cluster_avg,
    int*              num_updated)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= local_cols) return;

    int    best_label = 0;
    double best_dist  = 1e300;

    for (int k = 0; k < num_col_labels; k++) {
        double dist = 0.0;
        for (int row = 0; row < num_rows; row++) {
            double item = (double)matrix[row * local_cols + col];
            int    cl   = row_labels[row] * num_col_labels + k;
            double diff = cluster_avg[cl] - item;
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist  = dist;
            best_label = k;
        }
    }

    if (col_labels[col] != best_label) {
        col_labels[col] = best_label;
        atomicAdd(num_updated, 1);
    }
}

enum class SumVariant { ATOMIC, SHARED, WARP };

struct TunedConfig {
    int         cluster_sum;   // best block size
    SumVariant  sum_variant;   // best kernel variant
    int         divide_avg;
    int         row_dist;
    int         col_labels;
};

static float time_kernel_ms(std::function<void()> fn, int warmup=2, int runs=5) {
    for (int i = 0; i < warmup; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < runs; i++) fn();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / runs;
}

TunedConfig auto_tune(
    int rank,
    int num_rows, int local_cols,
    int num_row_labels, int num_col_labels, int num_clusters,
    float* d_matrix, label_type* d_row_labels, label_type* d_col_labels,
    double* d_cluster_avg, double* d_global_sum, int* d_global_count,
    double* d_partial_dist, int* d_cols_updated)
{
    const std::vector<int> candidates = {64, 128, 256, 512, 1024};
    TunedConfig best = {256, SumVariant::SHARED, 256, 256, 256};
    size_t shared_size = num_clusters * (sizeof(double) + sizeof(int));

    if (rank == 0) printf("\n[auto-tune] Benchmarking all kernel variants and block sizes...\n");

    // Tune kernel_cluster_sum variants
    // All three variants were benchmarked (global atomics, shared memory, warp shuffle) across a range of block sizes, 
    // and the best combination was selected. This is the most important kernel to tune, as it dominates the runtime and 
    // has multiple implementation strategies with different performance characteristics depending on contention patterns.

    {
        float best_ms = 1e9f;
        if (rank == 0) printf("\n  --- cluster_sum variants ---\n");

        for (int t : candidates) {
            int blocks = (num_rows * local_cols + t - 1) / t;

            // Variant A: global atomics
            float ms_a = time_kernel_ms([&]() {
                CUDA_CHECK(cudaMemset(d_global_sum,   0, num_clusters*sizeof(double)));
                CUDA_CHECK(cudaMemset(d_global_count, 0, num_clusters*sizeof(int)));
                kernel_cluster_sum<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels,
                    d_matrix, d_row_labels, d_col_labels, d_global_sum, d_global_count);
            });

            // Variant B: shared memory
            float ms_b = time_kernel_ms([&]() {
                CUDA_CHECK(cudaMemset(d_global_sum,   0, num_clusters*sizeof(double)));
                CUDA_CHECK(cudaMemset(d_global_count, 0, num_clusters*sizeof(int)));
                kernel_cluster_sum_shared<<<blocks, t, shared_size>>>(
                    num_rows, local_cols, num_col_labels, num_clusters,
                    d_matrix, d_row_labels, d_col_labels, d_global_sum, d_global_count);
            });

            // Variant C: warp shuffle
            float ms_c = time_kernel_ms([&]() {
                CUDA_CHECK(cudaMemset(d_global_sum,   0, num_clusters*sizeof(double)));
                CUDA_CHECK(cudaMemset(d_global_count, 0, num_clusters*sizeof(int)));
                kernel_cluster_sum_warp<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels, num_clusters,
                    d_matrix, d_row_labels, d_col_labels, d_global_sum, d_global_count);
            });

            if (rank == 0)
                printf("  block=%4d  atomic=%.3fms  shared=%.3fms  warp=%.3fms\n",
                       t, ms_a, ms_b, ms_c);

            // Pick best variant for this block size
            float best_at_t = ms_a; SumVariant var = SumVariant::ATOMIC;
            if (ms_b < best_at_t) { best_at_t = ms_b; var = SumVariant::SHARED; }
            //if (ms_c < best_at_t) { best_at_t = ms_c; var = SumVariant::WARP;   }

            if (best_at_t < best_ms) {
                best_ms            = best_at_t;
                best.cluster_sum   = t;
                best.sum_variant   = var;
            }
        }

        const char* vname = (best.sum_variant == SumVariant::ATOMIC) ? "atomic" :
                            (best.sum_variant == SumVariant::SHARED) ? "shared" : "warp";
        if (rank == 0)
            printf("  => best: block=%d variant=%s (%.3fms)\n\n", best.cluster_sum, vname, best_ms);
    }

    // Tune kernel_divide_avg
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  --- divide_avg ---\n");
        for (int t : candidates) {
            int blocks = (num_clusters + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                kernel_divide_avg<<<blocks, t>>>(
                    num_clusters, d_global_sum, d_global_count, d_cluster_avg);
            });
            if (rank == 0) printf("  block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.divide_avg = t; }
        }
        if (rank == 0) printf("  => best block=%d\n\n", best.divide_avg);
    }

    // Tune kernel_row_distances
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  --- row_distances ---\n");
        for (int t : candidates) {
            int blocks = (num_rows * num_row_labels + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                kernel_row_distances<<<blocks, t>>>(
                    num_rows, local_cols, num_row_labels, num_col_labels,
                    d_matrix, d_col_labels, d_cluster_avg, d_partial_dist);
            });
            if (rank == 0) printf("  block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.row_dist = t; }
        }
        if (rank == 0) printf("  => best block=%d\n\n", best.row_dist);
    }

    // Tune kernel_col_labels
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  --- col_labels ---\n");
        for (int t : candidates) {
            int blocks = (local_cols + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                CUDA_CHECK(cudaMemset(d_cols_updated, 0, sizeof(int)));
                kernel_col_labels<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels,
                    d_matrix, d_row_labels, d_col_labels,
                    d_cluster_avg, d_cols_updated);
            });
            if (rank == 0) printf("  block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.col_labels = t; }
        }
        if (rank == 0) printf("  => best block=%d\n\n", best.col_labels);
    }

    const char* vname = (best.sum_variant == SumVariant::ATOMIC) ? "atomic" :
                        (best.sum_variant == SumVariant::SHARED) ? "shared" : "warp";
    if (rank == 0) {
        printf("[auto-tune] Final config:\n");
        printf("  cluster_sum: block=%d variant=%s\n", best.cluster_sum, vname);
        printf("  divide_avg:  block=%d\n", best.divide_avg);
        printf("  row_dist:    block=%d\n", best.row_dist);
        printf("  col_labels:  block=%d\n\n", best.col_labels);
    }
    return best;
}

// Launch whichever cluster_sum variant was selected by auto-tuner
// shared_size was passed here to avoid computing it inside the kernel,
// which would be redundant if we're not using the shared variant.
void launch_cluster_sum(
    const TunedConfig& cfg, size_t shared_size,
    int num_rows, int local_cols, int num_col_labels, int num_clusters,
    const float* d_matrix, const label_type* d_row_labels,
    const label_type* d_col_labels, double* d_global_sum, int* d_global_count,
    cudaStream_t stream)
{
    int blocks = (num_rows * local_cols + cfg.cluster_sum - 1) / cfg.cluster_sum;
    switch (cfg.sum_variant) {
        case SumVariant::ATOMIC:
            kernel_cluster_sum<<<blocks, cfg.cluster_sum, 0, stream>>>(
                num_rows, local_cols, num_col_labels,
                d_matrix, d_row_labels, d_col_labels, d_global_sum, d_global_count);
            break;
        case SumVariant::SHARED:
            kernel_cluster_sum_shared<<<blocks, cfg.cluster_sum, shared_size, stream>>>(
                num_rows, local_cols, num_col_labels, num_clusters,
                d_matrix, d_row_labels, d_col_labels, d_global_sum, d_global_count);
            break;
        case SumVariant::WARP:
            kernel_cluster_sum_warp<<<blocks, cfg.cluster_sum, 0, stream>>>(
                num_rows, local_cols, num_col_labels, num_clusters,
                d_matrix, d_row_labels, d_col_labels, d_global_sum, d_global_count);
            break;
    }
}
// Main clustering loop: MPI distributes columns, GPU does the computation
void cluster_cuda(
    int rank,
    int nprocs,
    int num_rows,
    int num_cols,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,        // host
    label_type*  row_labels,          // host, replicated on all ranks
    label_type*  local_col_labels,    // host
    int          max_iterations)
{
    int    num_clusters = num_row_labels * num_col_labels;
    size_t shared_size  = num_clusters * (sizeof(double) + sizeof(int));

    // ── Device allocations ───────────────────────────────────────────────────
    float*      d_matrix;
    label_type* d_col_labels, *d_row_labels;
    double*     d_cluster_avg, *d_global_sum, *d_partial_dist;
    int*        d_global_count, *d_cols_updated;

    CUDA_CHECK(cudaMalloc(&d_matrix,       num_rows * local_cols        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_labels,   local_cols                   * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&d_row_labels,   num_rows                     * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&d_cluster_avg,  num_clusters                 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_global_sum,   num_clusters                 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_global_count, num_clusters                 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_dist, num_rows * num_row_labels    * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols_updated, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_matrix,     local_matrix,     num_rows*local_cols*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_labels, local_col_labels, local_cols*sizeof(label_type),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_labels, row_labels,       num_rows*sizeof(label_type),          cudaMemcpyHostToDevice));

    // Host buffers for MPI reductions
    double *h_local_sum, *h_global_sum, *h_local_dist, *h_global_dist;
    int    *h_local_count, *h_global_count;
    // Using cudaMallocHost to allocate pinned memory for faster GPU-CPU transfers during reductions
    CUDA_CHECK(cudaMallocHost(&h_local_sum,   num_clusters*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_local_count, num_clusters*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_global_sum,  num_clusters*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_global_count,num_clusters*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_local_dist,  num_rows*num_row_labels*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_global_dist, num_rows*num_row_labels*sizeof(double)));

    // Create separate CUDA streams for compute and transfer to allow overlapping communication and computation during MPI reductions
    cudaStream_t stream_compute, stream_transfer;
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_transfer));

    // Auto-tune kernels and block sizes before starting iterations
    TunedConfig cfg = auto_tune(
        rank, num_rows, local_cols, num_row_labels, num_col_labels, num_clusters,
        d_matrix, d_row_labels, d_col_labels,
        d_cluster_avg, d_global_sum, d_global_count,
        d_partial_dist, d_cols_updated);


    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {
/*
        // calculate cluster average
        CUDA_CHECK(cudaMemsetAsync(d_global_sum,   0, num_clusters*sizeof(double), stream_compute));
        CUDA_CHECK(cudaMemsetAsync(d_global_count, 0, num_clusters*sizeof(int),    stream_compute));

        // Launch the selected cluster_sum kernel variant using the auto-tuned configuration
        launch_cluster_sum(cfg, shared_size,
            num_rows, local_cols, num_col_labels, num_clusters,
            d_matrix, d_row_labels, d_col_labels,
            d_global_sum, d_global_count, stream_compute);

        // Copy local sums and counts back to host for MPI reduction
        CUDA_CHECK(cudaMemcpyAsync(h_local_sum,   d_global_sum,   num_clusters*sizeof(double), cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaMemcpyAsync(h_local_count, d_global_count, num_clusters*sizeof(int),    cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        MPI_Allreduce(h_local_sum,   h_global_sum,   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(h_local_count, h_global_count, num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        // Copy global sums and counts back to GPU for divide_avg kernel
        CUDA_CHECK(cudaMemcpyAsync(d_global_sum,   h_global_sum,   num_clusters*sizeof(double), cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaMemcpyAsync(d_global_count, h_global_count, num_clusters*sizeof(int),    cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));

        {
            int blocks = (num_clusters + cfg.divide_avg - 1) / cfg.divide_avg;
            kernel_divide_avg<<<blocks, cfg.divide_avg, 0, stream_compute>>>(
                num_clusters, d_global_sum, d_global_count, d_cluster_avg);
        }
*/
        //update_row_labels
        {
            int blocks = (num_rows * num_row_labels + cfg.row_dist - 1) / cfg.row_dist;
            kernel_row_distances<<<blocks, cfg.row_dist, 0, stream_compute>>>(
                num_rows, local_cols, num_row_labels, num_col_labels,
                d_matrix, d_col_labels, d_cluster_avg, d_partial_dist);
        }

        // Copy partial distances back to host for MPI reduction and label updates
        CUDA_CHECK(cudaMemcpyAsync(h_local_dist, d_partial_dist,
                                num_rows*num_row_labels*sizeof(double),
                                cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        MPI_Allreduce(h_local_dist, h_global_dist,
                    num_rows*num_row_labels, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        int    rows_updated   = 0;
        double total_dist_row = 0.0;
        for (int i = 0; i < num_rows; i++) {
            int best = 0; double bd = h_global_dist[i*num_row_labels];
            for (int k = 1; k < num_row_labels; k++) {
                double d = h_global_dist[i*num_row_labels+k];
                if (d < bd) { bd = d; best = k; }
            }
            if (row_labels[i] != best) { row_labels[i] = best; rows_updated++; }
            total_dist_row += bd;
        }

        // Upload updated row labels to GPU for next iteration
        CUDA_CHECK(cudaMemcpyAsync(d_row_labels, row_labels,
                                num_rows*sizeof(label_type),
                                cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));

        // After uploading updated row_labels to GPU (stream_transfer sync)...

        // Recompute cluster averages with updated row labels before updating col labels
        CUDA_CHECK(cudaMemsetAsync(d_global_sum,   0, num_clusters*sizeof(double), stream_compute));
        CUDA_CHECK(cudaMemsetAsync(d_global_count, 0, num_clusters*sizeof(int),    stream_compute));

        launch_cluster_sum(cfg, shared_size,
            num_rows, local_cols, num_col_labels, num_clusters,
            d_matrix, d_row_labels, d_col_labels,
            d_global_sum, d_global_count, stream_compute);

        CUDA_CHECK(cudaMemcpyAsync(h_local_sum,   d_global_sum,   num_clusters*sizeof(double), cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaMemcpyAsync(h_local_count, d_global_count, num_clusters*sizeof(int),    cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        MPI_Allreduce(h_local_sum,   h_global_sum,   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(h_local_count, h_global_count, num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_global_sum,   h_global_sum,   num_clusters*sizeof(double), cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaMemcpyAsync(d_global_count, h_global_count, num_clusters*sizeof(int),    cudaMemcpyHostToDevice, stream_transfer));
        CUDA_CHECK(cudaStreamSynchronize(stream_transfer));

        {
            int blocks = (num_clusters + cfg.divide_avg - 1) / cfg.divide_avg;
            kernel_divide_avg<<<blocks, cfg.divide_avg, 0, stream_compute>>>(
                num_clusters, d_global_sum, d_global_count, d_cluster_avg);
        }

        // STEP 3: update_col_labels
        CUDA_CHECK(cudaMemsetAsync(d_cols_updated, 0, sizeof(int), stream_compute));
        {
            int blocks = (local_cols + cfg.col_labels - 1) / cfg.col_labels;
            kernel_col_labels<<<blocks, cfg.col_labels, 0, stream_compute>>>(
                num_rows, local_cols, num_col_labels,
                d_matrix, d_row_labels, d_col_labels,
                d_cluster_avg, d_cols_updated);
        }
        int local_cols_updated = 0;
        CUDA_CHECK(cudaMemcpyAsync(&local_cols_updated, d_cols_updated,
                                sizeof(int), cudaMemcpyDeviceToHost, stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_compute));

        int global_cols_updated = 0;
        MPI_Allreduce(&local_cols_updated, &global_cols_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        iteration++;
        int    num_updated  = rows_updated + global_cols_updated;
        double average_dist = total_dist_row / (double)(num_rows * num_cols);

            if (rank == 0) {
                std::cout << "iteration " << iteration << ": " << num_updated
                        << " labels were updated, average error is " << average_dist
                        << "\n";
            }

            if (num_updated == 0) break;
    }

    auto after = std::chrono::high_resolution_clock::now();
    double time_seconds = std::chrono::duration<double>(after - before).count();


    if (rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: " << (time_seconds / iteration)
                << " seconds\n";
    }

    CUDA_CHECK(cudaMemcpy(local_col_labels, d_col_labels,local_cols * sizeof(label_type), cudaMemcpyDeviceToHost));


    cudaFree(d_matrix); cudaFree(d_col_labels); cudaFree(d_row_labels);
    cudaFree(d_cluster_avg); cudaFree(d_global_sum); cudaFree(d_global_count);
    cudaFree(d_partial_dist); cudaFree(d_cols_updated);
    cudaFreeHost(h_local_sum); cudaFreeHost(h_local_count);
    cudaFreeHost(h_global_sum); cudaFreeHost(h_global_count);
    cudaFreeHost(h_local_dist); cudaFreeHost(h_global_dist);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_transfer);
}

int main(int argc, const char* argv[]) {
    MPI_Init(nullptr, nullptr);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    auto wall_start = std::chrono::high_resolution_clock::now();

    // Parse arguments on rank 0 only
    std::string output_file;
    std::vector<float>      matrix;
    std::vector<label_type> row_labels, col_labels;
    int num_rows = 0, num_cols = 0;
    int num_row_labels = 0, num_col_labels = 0;
    int max_iter = 0;

    if (rank == 0) {
        if (!parse_arguments(
                argc, argv,
                &num_rows, &num_cols,
                &num_row_labels, &num_col_labels,
                &matrix, &row_labels, &col_labels,
                &output_file, &max_iter)) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast scalar parameters
    MPI_Bcast(&num_rows,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter,       1, MPI_INT, 0, MPI_COMM_WORLD);

    int fname_len = (rank == 0) ? int(output_file.size()) : 0;
    MPI_Bcast(&fname_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) output_file.resize(fname_len);
    MPI_Bcast(output_file.data(), fname_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Compute column distribution 
    std::vector<int> sendcounts(nprocs), displs(nprocs);
    int base = num_cols / nprocs;
    int rem  = num_cols % nprocs;
    for (int p = 0; p < nprocs; p++) {
        sendcounts[p] = base + (p < rem ? 1 : 0);
        displs[p]     = (p == 0) ? 0 : displs[p-1] + sendcounts[p-1];
    }
    int local_cols = sendcounts[rank];

    // Distribute labels
    std::vector<label_type> local_col_labels(local_cols);

    if (rank != 0) row_labels.resize(num_rows);
    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatterv(
        (rank == 0) ? col_labels.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_INT,
        local_col_labels.data(), local_cols, MPI_INT,
        0, MPI_COMM_WORLD);

    // Distribute matrix (row by row)
    std::vector<float> local_matrix(num_rows * local_cols);
    for (int i = 0; i < num_rows; i++) {
        const float* row_src = (rank == 0) ? (matrix.data() + i * num_cols) : nullptr;
        float*       row_dst = local_matrix.data() + i * local_cols;
        MPI_Scatterv(
            row_src, sendcounts.data(), displs.data(), MPI_FLOAT,
            row_dst, local_cols, MPI_FLOAT,
            0, MPI_COMM_WORLD);
    }

    // Free full matrix on rank 0
    if (rank == 0) { matrix.clear(); matrix.shrink_to_fit(); }

    // Run CUDA co-clustering
    cluster_cuda(
        rank, nprocs,
        num_rows, num_cols, local_cols,
        num_row_labels, num_col_labels,
        local_matrix.data(),
        row_labels.data(),
        local_col_labels.data(),
        max_iter);

    // Gather col_labels back to rank 0 
    if (rank == 0) col_labels.resize(num_cols);

    MPI_Gatherv(
        local_col_labels.data(), local_cols, MPI_INT,
        (rank == 0) ? col_labels.data() : nullptr,
        sendcounts.data(), displs.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    // Write output
    if (rank == 0) {
        write_labels(
            output_file,
            num_rows, num_cols,
            row_labels.data(),
            col_labels.data());
    }

    // Report total wall time
    auto wall_end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        double time_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
        std::cout << "total execution time: " << time_seconds << " seconds\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

