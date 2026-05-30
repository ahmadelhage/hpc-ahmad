#include <chrono>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>

#include "common.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// VARIANT A: cluster sum using global atomics (baseline reference)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_cluster_sum_atomic(
    int num_rows, int local_cols, int num_col_labels,
    const float* matrix, const label_type* row_labels,
    const label_type* col_labels, double* local_sum, int* local_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows * local_cols) return;
    int row     = idx / local_cols;
    int col     = idx % local_cols;
    double item = (double)matrix[row * local_cols + col];
    int cluster = row_labels[row] * num_col_labels + col_labels[col];
    atomicAdd(&local_sum[cluster],   item);
    atomicAdd(&local_count[cluster], 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// VARIANT B: cluster sum using shared memory
//   Each block accumulates into block-local shared memory, reducing global
//   atomic contention by up to blockDim.x times.
//   Dynamic shared mem layout: [num_clusters doubles | num_clusters ints]
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_cluster_sum_shared(
    int num_rows, int local_cols, int num_col_labels, int num_clusters,
    const float* matrix, const label_type* row_labels,
    const label_type* col_labels, double* global_sum, int* global_count)
{
    extern __shared__ char smem[];
    double* s_sum   = reinterpret_cast<double*>(smem);
    int*    s_count = reinterpret_cast<int*>(s_sum + num_clusters);

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        s_sum[i]   = 0.0;
        s_count[i] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_rows * local_cols) {
        int row     = idx / local_cols;
        int col     = idx % local_cols;
        double item = (double)matrix[row * local_cols + col];
        int cluster = row_labels[row] * num_col_labels + col_labels[col];
        atomicAdd(&s_sum[cluster],   item);
        atomicAdd(&s_count[cluster], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        if (s_sum[i]   != 0.0) atomicAdd(&global_sum[i],   s_sum[i]);
        if (s_count[i] != 0)   atomicAdd(&global_count[i], s_count[i]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// VARIANT C: cluster sum using warp shuffle
//   Uses __shfl_down_sync to exchange values in registers between warp lanes.
//   Benchmarked for reference — not selected by auto-tuner due to correctness
//   issues with non-uniform cluster distributions across warp lanes.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_cluster_sum_warp(
    int num_rows, int local_cols, int num_col_labels,
    const float* matrix, const label_type* row_labels,
    const label_type* col_labels, double* global_sum, int* global_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows * local_cols) return;
    int row     = idx / local_cols;
    int col     = idx % local_cols;
    double item = (double)matrix[row * local_cols + col];
    int cluster = row_labels[row] * num_col_labels + col_labels[col];
    unsigned mask = 0xffffffff;
    int lane = threadIdx.x & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_val     = __shfl_down_sync(mask, item,    offset);
        int    other_cluster = __shfl_down_sync(mask, cluster, offset);
        if (other_cluster == cluster) item += other_val;
    }
    int prev_cluster = __shfl_up_sync(mask, cluster, 1);
    if (lane == 0 || prev_cluster != cluster) {
        atomicAdd(&global_sum[cluster],   item);
        atomicAdd(&global_count[cluster], 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL: divide global sums by counts to get cluster averages
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_divide_avg(
    int num_clusters, const double* global_sum,
    const int* global_count, double* cluster_avg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clusters) return;
    cluster_avg[idx] = global_sum[idx] / (double)global_count[idx];
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL: compute partial row-label distances (one thread per row x candidate)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_row_distances(
    int num_rows, int local_cols, int num_row_labels, int num_col_labels,
    const float* matrix, const label_type* col_labels,
    const double* cluster_avg, double* partial_dist)
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
        dist += diff * diff;
    }
    partial_dist[row * num_row_labels + k] = dist;
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL: select best row label on GPU (runs after MPI Allreduce)
//   Each thread handles one row. Scans global_dist to find the minimum.
//   Writes new label to d_row_labels and atomicAdds to rows_updated.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_select_row_labels(
    int num_rows, int num_row_labels,
    const double* global_dist,
    label_type*   row_labels,
    int*          rows_updated,
    double*       total_dist_out)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int    best      = 0;
    double best_dist = global_dist[row * num_row_labels];
    for (int k = 1; k < num_row_labels; k++) {
        double d = global_dist[row * num_row_labels + k];
        if (d < best_dist) { best_dist = d; best = k; }
    }

    if (row_labels[row] != best) {
        row_labels[row] = best;
        atomicAdd(rows_updated, 1);
    }
    // Accumulate total distance using double atomicAdd
    // (supported on sm_60+ which A4000 Ada satisfies)
    atomicAdd(total_dist_out, best_dist);
}

// ─────────────────────────────────────────────────────────────────────────────
// KERNEL: update column labels (one thread per local column)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_col_labels(
    int num_rows, int local_cols, int num_col_labels,
    const float* matrix, const label_type* row_labels,
    label_type* col_labels, const double* cluster_avg, int* num_updated)
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
        if (dist < best_dist) { best_dist = dist; best_label = k; }
    }
    if (col_labels[col] != best_label) {
        col_labels[col] = best_label;
        atomicAdd(num_updated, 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTO-TUNING: benchmark all variants and block sizes, pick fastest
// ─────────────────────────────────────────────────────────────────────────────
enum class SumVariant { ATOMIC, SHARED };

struct TunedSizes {
    int        cluster_sum;
    SumVariant sum_variant;
    int        divide_avg;
    int        row_dist;
    int        row_select;
    int        col_labels;
};

static float time_kernel_ms(
    std::function<void()> fn, int warmup = 2, int runs = 5)
{
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

TunedSizes auto_tune(
    int rank,
    int num_rows, int local_cols,
    int num_row_labels, int num_col_labels, int num_clusters,
    float* d_matrix, label_type* d_row_labels, label_type* d_col_labels,
    double* d_cluster_avg, double* d_local_sum, int* d_local_count,
    double* d_partial_dist, int* d_cols_updated,
    double* d_global_dist, int* d_rows_updated, double* d_total_dist)
{
    const std::vector<int> candidates = {64, 128, 256, 512, 1024};
    TunedSizes best = {256, SumVariant::ATOMIC, 256, 256, 256, 256};
    size_t shared_size = num_clusters * (sizeof(double) + sizeof(int));

    if (rank == 0) printf("\n[auto-tune] Benchmarking all kernel variants...\n");

    // cluster_sum: all three variants, select only atomic or shared
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  cluster_sum (3 variants):\n");
        if (rank == 0) printf("  %6s  %10s  %10s  %10s\n","block","atomic","shared","warp(ref)");
        for (int t : candidates) {
            int blocks = (num_rows * local_cols + t - 1) / t;
            float ms_a = time_kernel_ms([&]() {
                cudaMemset(d_local_sum,   0, num_clusters*sizeof(double));
                cudaMemset(d_local_count, 0, num_clusters*sizeof(int));
                kernel_cluster_sum_atomic<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels,
                    d_matrix, d_row_labels, d_col_labels, d_local_sum, d_local_count);
            });
            float ms_s = time_kernel_ms([&]() {
                cudaMemset(d_local_sum,   0, num_clusters*sizeof(double));
                cudaMemset(d_local_count, 0, num_clusters*sizeof(int));
                kernel_cluster_sum_shared<<<blocks, t, shared_size>>>(
                    num_rows, local_cols, num_col_labels, num_clusters,
                    d_matrix, d_row_labels, d_col_labels, d_local_sum, d_local_count);
            });
            float ms_w = time_kernel_ms([&]() {
                cudaMemset(d_local_sum,   0, num_clusters*sizeof(double));
                cudaMemset(d_local_count, 0, num_clusters*sizeof(int));
                kernel_cluster_sum_warp<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels,
                    d_matrix, d_row_labels, d_col_labels, d_local_sum, d_local_count);
            });
            if (rank == 0) printf("  %6d  %8.3fms  %8.3fms  %8.3fms\n", t, ms_a, ms_s, ms_w);
            if (ms_a < best_ms) { best_ms = ms_a; best.cluster_sum = t; best.sum_variant = SumVariant::ATOMIC; }
            if (ms_s < best_ms) { best_ms = ms_s; best.cluster_sum = t; best.sum_variant = SumVariant::SHARED; }
        }
        const char* v = (best.sum_variant == SumVariant::SHARED) ? "shared" : "atomic";
        if (rank == 0) printf("  => best: block=%d variant=%s\n", best.cluster_sum, v);
    }

    // divide_avg
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  divide_avg:\n");
        for (int t : candidates) {
            int blocks = (num_clusters + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                kernel_divide_avg<<<blocks, t>>>(
                    num_clusters, d_local_sum, d_local_count, d_cluster_avg);
            });
            if (rank == 0) printf("    block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.divide_avg = t; }
        }
        if (rank == 0) printf("  => best: %d\n", best.divide_avg);
    }

    // row_distances
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  row_distances:\n");
        for (int t : candidates) {
            int blocks = (num_rows * num_row_labels + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                kernel_row_distances<<<blocks, t>>>(
                    num_rows, local_cols, num_row_labels, num_col_labels,
                    d_matrix, d_col_labels, d_cluster_avg, d_partial_dist);
            });
            if (rank == 0) printf("    block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.row_dist = t; }
        }
        if (rank == 0) printf("  => best: %d\n", best.row_dist);
    }

    // row_select (new kernel)
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  row_select:\n");
        for (int t : candidates) {
            int blocks = (num_rows + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                cudaMemset(d_rows_updated, 0, sizeof(int));
                cudaMemset(d_total_dist,   0, sizeof(double));
                kernel_select_row_labels<<<blocks, t>>>(
                    num_rows, num_row_labels,
                    d_global_dist, d_row_labels,
                    d_rows_updated, d_total_dist);
            });
            if (rank == 0) printf("    block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.row_select = t; }
        }
        if (rank == 0) printf("  => best: %d\n", best.row_select);
    }

    // col_labels
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  col_labels:\n");
        for (int t : candidates) {
            int blocks = (local_cols + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                cudaMemset(d_cols_updated, 0, sizeof(int));
                kernel_col_labels<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels,
                    d_matrix, d_row_labels, d_col_labels,
                    d_cluster_avg, d_cols_updated);
            });
            if (rank == 0) printf("    block=%4d  %.3fms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.col_labels = t; }
        }
        if (rank == 0) printf("  => best: %d\n", best.col_labels);
    }

    const char* v = (best.sum_variant == SumVariant::SHARED) ? "shared" : "atomic";
    if (rank == 0)
        printf("[auto-tune] Config: cluster_sum=%d(%s) divide_avg=%d row_dist=%d row_select=%d col_labels=%d\n\n",
               best.cluster_sum, v, best.divide_avg, best.row_dist, best.row_select, best.col_labels);

    // Reset all scratch arrays
    CUDA_CHECK(cudaMemset(d_local_sum,    0, num_clusters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_count,  0, num_clusters * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cluster_avg,  0, num_clusters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cols_updated, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_rows_updated, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_total_dist,   0, sizeof(double)));

    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: launch whichever cluster_sum variant was chosen
// ─────────────────────────────────────────────────────────────────────────────
void launch_cluster_sum(
    const TunedSizes& tuned, size_t shared_size,
    int num_rows, int local_cols, int num_col_labels, int num_clusters,
    const float* d_matrix, const label_type* d_row_labels,
    const label_type* d_col_labels, double* d_local_sum, int* d_local_count,
    cudaStream_t stream)
{
    int blocks = (num_rows * local_cols + tuned.cluster_sum - 1) / tuned.cluster_sum;
    if (tuned.sum_variant == SumVariant::SHARED) {
        kernel_cluster_sum_shared<<<blocks, tuned.cluster_sum, shared_size, stream>>>(
            num_rows, local_cols, num_col_labels, num_clusters,
            d_matrix, d_row_labels, d_col_labels, d_local_sum, d_local_count);
    } else {
        kernel_cluster_sum_atomic<<<blocks, tuned.cluster_sum, 0, stream>>>(
            num_rows, local_cols, num_col_labels,
            d_matrix, d_row_labels, d_col_labels, d_local_sum, d_local_count);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU CONTEXT: all device pointers and streams in one struct for clean passing
// ─────────────────────────────────────────────────────────────────────────────
struct GpuContext {
    float*      d_matrix;
    label_type* d_col_labels, *d_row_labels;
    double*     d_cluster_avg, *d_local_sum, *d_partial_dist;
    double*     d_global_dist;
    int*        d_local_count, *d_cols_updated, *d_rows_updated;
    double*     d_total_dist;
    double      *h_local_sum,  *h_global_sum,  *h_local_dist, *h_global_dist;
    int         *h_local_count,*h_global_count;
    cudaStream_t stream_compute, stream_transfer;
    size_t       shared_size;
};

// ─────────────────────────────────────────────────────────────────────────────
// STEP 1: calculate_cluster_average
//   Launches cluster_sum kernel, copies partial sums to host, does
//   MPI_Allreduce, copies back, then runs divide_avg kernel.
// ─────────────────────────────────────────────────────────────────────────────
void calculate_cluster_average(
    int num_clusters, const TunedSizes& tuned,
    int num_rows, int local_cols, int num_col_labels,
    GpuContext& g)
{
    CUDA_CHECK(cudaMemsetAsync(g.d_local_sum,   0, num_clusters*sizeof(double), g.stream_compute));
    CUDA_CHECK(cudaMemsetAsync(g.d_local_count, 0, num_clusters*sizeof(int),    g.stream_compute));

    launch_cluster_sum(tuned, g.shared_size,
        num_rows, local_cols, num_col_labels, num_clusters,
        g.d_matrix, g.d_row_labels, g.d_col_labels,
        g.d_local_sum, g.d_local_count, g.stream_compute);

    CUDA_CHECK(cudaMemcpyAsync(g.h_local_sum,   g.d_local_sum,   num_clusters*sizeof(double), cudaMemcpyDeviceToHost, g.stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(g.h_local_count, g.d_local_count, num_clusters*sizeof(int),    cudaMemcpyDeviceToHost, g.stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(g.stream_compute));

    MPI_Allreduce(g.h_local_sum,   g.h_global_sum,   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(g.h_local_count, g.h_global_count, num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

    CUDA_CHECK(cudaMemcpyAsync(g.d_local_sum,   g.h_global_sum,   num_clusters*sizeof(double), cudaMemcpyHostToDevice, g.stream_transfer));
    CUDA_CHECK(cudaMemcpyAsync(g.d_local_count, g.h_global_count, num_clusters*sizeof(int),    cudaMemcpyHostToDevice, g.stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(g.stream_transfer));

    int blocks = (num_clusters + tuned.divide_avg - 1) / tuned.divide_avg;
    kernel_divide_avg<<<blocks, tuned.divide_avg, 0, g.stream_compute>>>(
        num_clusters, g.d_local_sum, g.d_local_count, g.d_cluster_avg);
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 2: update_row_labels
//   Launches row_distances kernel, copies partial distances to host,
//   does MPI_Allreduce, copies global distances back to GPU, then
//   kernel_select_row_labels runs entirely on GPU to pick best labels.
//   Returns {rows_updated, total_dist}.
// ─────────────────────────────────────────────────────────────────────────────
std::pair<int, double> update_row_labels(
    int num_rows, int local_cols, int num_row_labels, int num_col_labels,
    const TunedSizes& tuned, GpuContext& g)
{
    int work = num_rows * num_row_labels;
    int blocks = (work + tuned.row_dist - 1) / tuned.row_dist;
    kernel_row_distances<<<blocks, tuned.row_dist, 0, g.stream_compute>>>(
        num_rows, local_cols, num_row_labels, num_col_labels,
        g.d_matrix, g.d_col_labels, g.d_cluster_avg, g.d_partial_dist);

    CUDA_CHECK(cudaMemcpyAsync(g.h_local_dist, g.d_partial_dist,
                               work*sizeof(double), cudaMemcpyDeviceToHost, g.stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(g.stream_compute));

    MPI_Allreduce(g.h_local_dist, g.h_global_dist,
                  work, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // Copy global distances back to GPU for kernel_select_row_labels
    CUDA_CHECK(cudaMemcpyAsync(g.d_global_dist, g.h_global_dist,
                               work*sizeof(double), cudaMemcpyHostToDevice, g.stream_transfer));
    CUDA_CHECK(cudaStreamSynchronize(g.stream_transfer));

    // Select best row labels entirely on GPU
    CUDA_CHECK(cudaMemsetAsync(g.d_rows_updated, 0, sizeof(int),    g.stream_compute));
    CUDA_CHECK(cudaMemsetAsync(g.d_total_dist,   0, sizeof(double), g.stream_compute));

    int sel_blocks = (num_rows + tuned.row_select - 1) / tuned.row_select;
    kernel_select_row_labels<<<sel_blocks, tuned.row_select, 0, g.stream_compute>>>(
        num_rows, num_row_labels,
        g.d_global_dist, g.d_row_labels,
        g.d_rows_updated, g.d_total_dist);

    // Copy results back to host (tiny: 1 int + 1 double)
    int    rows_updated = 0;
    double total_dist   = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&rows_updated, g.d_rows_updated, sizeof(int),    cudaMemcpyDeviceToHost, g.stream_compute));
    CUDA_CHECK(cudaMemcpyAsync(&total_dist,   g.d_total_dist,   sizeof(double), cudaMemcpyDeviceToHost, g.stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(g.stream_compute));

    return {rows_updated, total_dist};
}

// ─────────────────────────────────────────────────────────────────────────────
// STEP 3: update_col_labels
//   Fully local — no MPI needed. Returns {cols_updated_globally, 0}.
// ─────────────────────────────────────────────────────────────────────────────
std::pair<int, double> update_col_labels(
    int num_rows, int local_cols, int num_col_labels,
    const TunedSizes& tuned, GpuContext& g)
{
    CUDA_CHECK(cudaMemsetAsync(g.d_cols_updated, 0, sizeof(int), g.stream_compute));

    int blocks = (local_cols + tuned.col_labels - 1) / tuned.col_labels;
    kernel_col_labels<<<blocks, tuned.col_labels, 0, g.stream_compute>>>(
        num_rows, local_cols, num_col_labels,
        g.d_matrix, g.d_row_labels, g.d_col_labels,
        g.d_cluster_avg, g.d_cols_updated);

    int local_cols_updated = 0;
    CUDA_CHECK(cudaMemcpyAsync(&local_cols_updated, g.d_cols_updated,
                               sizeof(int), cudaMemcpyDeviceToHost, g.stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(g.stream_compute));

    int global_cols_updated = 0;
    MPI_Allreduce(&local_cols_updated, &global_cols_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return {global_cols_updated, 0.0};
}

// ─────────────────────────────────────────────────────────────────────────────
// Main clustering loop
// ─────────────────────────────────────────────────────────────────────────────
void cluster_cuda(
    int rank, int nprocs,
    int num_rows, int num_cols, int local_cols,
    int num_row_labels, int num_col_labels,
    const float* local_matrix,
    label_type*  row_labels,
    label_type*  local_col_labels,
    int max_iterations)
{
    int    num_clusters = num_row_labels * num_col_labels;
    int    work_rows    = num_rows * num_row_labels;

    GpuContext g;
    g.shared_size = num_clusters * (sizeof(double) + sizeof(int));

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&g.d_matrix,       num_rows * local_cols  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g.d_col_labels,   local_cols             * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&g.d_row_labels,   num_rows               * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&g.d_cluster_avg,  num_clusters           * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_local_sum,    num_clusters           * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_local_count,  num_clusters           * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_partial_dist, work_rows              * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_global_dist,  work_rows              * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&g.d_cols_updated, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_rows_updated, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_total_dist,   sizeof(double)));

    CUDA_CHECK(cudaMemcpy(g.d_matrix,     local_matrix,     num_rows*local_cols*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_col_labels, local_col_labels, local_cols*sizeof(label_type),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_row_labels, row_labels,       num_rows*sizeof(label_type),          cudaMemcpyHostToDevice));

    // Allocate pinned host memory
    CUDA_CHECK(cudaMallocHost(&g.h_local_sum,    num_clusters*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&g.h_local_count,  num_clusters*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&g.h_global_sum,   num_clusters*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&g.h_global_count, num_clusters*sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&g.h_local_dist,   work_rows*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&g.h_global_dist,  work_rows*sizeof(double)));

    CUDA_CHECK(cudaStreamCreate(&g.stream_compute));
    CUDA_CHECK(cudaStreamCreate(&g.stream_transfer));

    // Auto-tune
    TunedSizes tuned = auto_tune(
        rank, num_rows, local_cols, num_row_labels, num_col_labels, num_clusters,
        g.d_matrix, g.d_row_labels, g.d_col_labels, g.d_cluster_avg,
        g.d_local_sum, g.d_local_count, g.d_partial_dist, g.d_cols_updated,
        g.d_global_dist, g.d_rows_updated, g.d_total_dist);

    // Re-upload labels after tuning
    CUDA_CHECK(cudaMemcpy(g.d_col_labels, local_col_labels, local_cols*sizeof(label_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g.d_row_labels, row_labels,       num_rows*sizeof(label_type),   cudaMemcpyHostToDevice));

    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {

        calculate_cluster_average(num_clusters, tuned,
            num_rows, local_cols, num_col_labels, g);

        auto [rows_updated, total_dist_row] = update_row_labels(
            num_rows, local_cols, num_row_labels, num_col_labels, tuned, g);

        auto [global_cols_updated, unused] = update_col_labels(
            num_rows, local_cols, num_col_labels, tuned, g);

        iteration++;
        int    num_updated  = rows_updated + global_cols_updated;
        double average_dist = total_dist_row / (double)(num_rows * num_cols);

        if (rank == 0)
            std::cout << "iteration " << iteration << ": " << num_updated
                      << " labels were updated, average error is " << average_dist << "\n";

        if (num_updated == 0) break;
    }

    auto after = std::chrono::high_resolution_clock::now();
    double time_seconds = std::chrono::duration<double>(after - before).count();
    if (rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: " << (time_seconds/iteration) << " seconds\n";
    }

    CUDA_CHECK(cudaMemcpy(local_col_labels, g.d_col_labels,
                          local_cols*sizeof(label_type), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaFree(g.d_matrix);    cudaFree(g.d_col_labels); cudaFree(g.d_row_labels);
    cudaFree(g.d_cluster_avg); cudaFree(g.d_local_sum); cudaFree(g.d_local_count);
    cudaFree(g.d_partial_dist); cudaFree(g.d_global_dist);
    cudaFree(g.d_cols_updated); cudaFree(g.d_rows_updated); cudaFree(g.d_total_dist);

    // Free pinned memory
    cudaFreeHost(g.h_local_sum);  cudaFreeHost(g.h_local_count);
    cudaFreeHost(g.h_global_sum); cudaFreeHost(g.h_global_count);
    cudaFreeHost(g.h_local_dist); cudaFreeHost(g.h_global_dist);

    cudaStreamDestroy(g.stream_compute);
    cudaStreamDestroy(g.stream_transfer);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, const char* argv[]) {
    MPI_Init(nullptr, nullptr);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    auto wall_start = std::chrono::high_resolution_clock::now();

    std::string output_file;
    std::vector<float>      matrix;
    std::vector<label_type> row_labels, col_labels;
    int num_rows=0, num_cols=0, num_row_labels=0, num_col_labels=0, max_iter=0;

    if (rank == 0) {
        if (!parse_arguments(argc, argv, &num_rows, &num_cols,
                             &num_row_labels, &num_col_labels,
                             &matrix, &row_labels, &col_labels,
                             &output_file, &max_iter))
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    MPI_Bcast(&num_rows,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter,       1, MPI_INT, 0, MPI_COMM_WORLD);

    int fname_len = (rank==0) ? int(output_file.size()) : 0;
    MPI_Bcast(&fname_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) output_file.resize(fname_len);
    MPI_Bcast(output_file.data(), fname_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    std::vector<int> sendcounts(nprocs), displs(nprocs);
    int base = num_cols/nprocs, rem = num_cols%nprocs;
    for (int p = 0; p < nprocs; p++) {
        sendcounts[p] = base + (p < rem ? 1 : 0);
        displs[p]     = (p==0) ? 0 : displs[p-1]+sendcounts[p-1];
    }
    int local_cols = sendcounts[rank];

    std::vector<label_type> local_col_labels(local_cols);
    if (rank != 0) row_labels.resize(num_rows);
    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv((rank==0)?col_labels.data():nullptr,
                 sendcounts.data(), displs.data(), MPI_INT,
                 local_col_labels.data(), local_cols, MPI_INT,
                 0, MPI_COMM_WORLD);

    std::vector<float> local_matrix(num_rows * local_cols);
    for (int i = 0; i < num_rows; i++) {
        const float* src = (rank==0) ? (matrix.data()+i*num_cols) : nullptr;
        MPI_Scatterv(src, sendcounts.data(), displs.data(), MPI_FLOAT,
                     local_matrix.data()+i*local_cols, local_cols, MPI_FLOAT,
                     0, MPI_COMM_WORLD);
    }
    if (rank==0) { matrix.clear(); matrix.shrink_to_fit(); }

    cluster_cuda(rank, nprocs, num_rows, num_cols, local_cols,
                 num_row_labels, num_col_labels,
                 local_matrix.data(), row_labels.data(),
                 local_col_labels.data(), max_iter);

    if (rank==0) col_labels.resize(num_cols);
    MPI_Gatherv(local_col_labels.data(), local_cols, MPI_INT,
                (rank==0)?col_labels.data():nullptr,
                sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank==0)
        write_labels(output_file, num_rows, num_cols, row_labels.data(), col_labels.data());

    auto wall_end = std::chrono::high_resolution_clock::now();
    if (rank==0) {
        double t = std::chrono::duration<double>(wall_end-wall_start).count();
        std::cout << "total execution time: " << t << " seconds\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
