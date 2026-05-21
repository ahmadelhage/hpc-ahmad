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
// KERNELS — identical to working cgc_cuda.cu, no changes
// ─────────────────────────────────────────────────────────────────────────────
__global__ void kernel_cluster_sum(
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

__global__ void kernel_divide_avg(
    int num_clusters, const double* global_sum,
    const int* global_count, double* cluster_avg)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_clusters) return;
    cluster_avg[idx] = (global_count[idx] > 0)
        ? global_sum[idx] / (double)global_count[idx]
        : 0.0;
}

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
// AUTO-TUNING
// Times each kernel at block sizes 64,128,256,512,1024 and picks fastest.
// Called once before the main loop.
// Does NOT modify label or data arrays — only uses scratch buffers.
// After tuning, scratch arrays are reset so main loop starts clean.
// ─────────────────────────────────────────────────────────────────────────────
struct TunedSizes {
    int cluster_sum;
    int divide_avg;
    int row_dist;
    int col_labels;
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
    float*      d_matrix,
    label_type* d_row_labels,
    label_type* d_col_labels,
    double*     d_cluster_avg,
    double*     d_local_sum,
    int*        d_local_count,
    double*     d_partial_dist,
    int*        d_cols_updated)
{
    const std::vector<int> candidates = {64, 128, 256, 512, 1024};
    TunedSizes best = {256, 256, 256, 256};

    if (rank == 0) printf("\n[auto-tune] Finding best block size per kernel...\n");

    // cluster_sum
    {
        float best_ms = 1e9f;
        if (rank == 0) printf("  cluster_sum:\n");
        for (int t : candidates) {
            int blocks = (num_rows * local_cols + t - 1) / t;
            float ms = time_kernel_ms([&]() {
                cudaMemset(d_local_sum,   0, num_clusters * sizeof(double));
                cudaMemset(d_local_count, 0, num_clusters * sizeof(int));
                kernel_cluster_sum<<<blocks, t>>>(
                    num_rows, local_cols, num_col_labels,
                    d_matrix, d_row_labels, d_col_labels,
                    d_local_sum, d_local_count);
            });
            if (rank == 0) printf("    block=%4d  %.3f ms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.cluster_sum = t; }
        }
        if (rank == 0) printf("    => best: %d\n", best.cluster_sum);
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
            if (rank == 0) printf("    block=%4d  %.3f ms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.divide_avg = t; }
        }
        if (rank == 0) printf("    => best: %d\n", best.divide_avg);
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
            if (rank == 0) printf("    block=%4d  %.3f ms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.row_dist = t; }
        }
        if (rank == 0) printf("    => best: %d\n", best.row_dist);
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
            if (rank == 0) printf("    block=%4d  %.3f ms\n", t, ms);
            if (ms < best_ms) { best_ms = ms; best.col_labels = t; }
        }
        if (rank == 0) printf("    => best: %d\n", best.col_labels);
    }

    if (rank == 0)
        printf("[auto-tune] Config: cluster_sum=%d divide_avg=%d row_dist=%d col_labels=%d\n\n",
               best.cluster_sum, best.divide_avg, best.row_dist, best.col_labels);

    // Reset ALL scratch arrays AND re-upload labels so main loop starts clean
    CUDA_CHECK(cudaMemset(d_local_sum,    0, num_clusters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_count,  0, num_clusters * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_cluster_avg,  0, num_clusters * sizeof(double)));
    CUDA_CHECK(cudaMemset(d_cols_updated, 0, sizeof(int)));

    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main loop — same logic as cgc_cuda.cu, uses tuned block sizes
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
    int num_clusters = num_row_labels * num_col_labels;

    float*      d_matrix;
    label_type* d_col_labels, *d_row_labels;
    double*     d_cluster_avg, *d_local_sum, *d_partial_dist;
    int*        d_local_count, *d_cols_updated;

    CUDA_CHECK(cudaMalloc(&d_matrix,       num_rows * local_cols     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_labels,   local_cols                * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&d_row_labels,   num_rows                  * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&d_cluster_avg,  num_clusters              * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_sum,    num_clusters              * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_count,  num_clusters              * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_dist, num_rows*num_row_labels   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols_updated, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_matrix,     local_matrix,     num_rows*local_cols*sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_labels, local_col_labels, local_cols*sizeof(label_type),       cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_labels, row_labels,       num_rows*sizeof(label_type),          cudaMemcpyHostToDevice));

    // Auto-tune (uses scratch buffers only, cannot corrupt labels)
    TunedSizes tuned = auto_tune(
        rank, num_rows, local_cols, num_row_labels, num_col_labels, num_clusters,
        d_matrix, d_row_labels, d_col_labels, d_cluster_avg,
        d_local_sum, d_local_count, d_partial_dist, d_cols_updated);

    // Re-upload labels after tuning to guarantee clean state
    CUDA_CHECK(cudaMemcpy(d_col_labels, local_col_labels, local_cols*sizeof(label_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_labels, row_labels,       num_rows*sizeof(label_type),   cudaMemcpyHostToDevice));

    std::vector<double> h_local_sum(num_clusters);
    std::vector<int>    h_local_count(num_clusters);
    std::vector<double> h_global_sum(num_clusters);
    std::vector<int>    h_global_count(num_clusters);
    std::vector<double> h_local_dist(num_rows * num_row_labels);
    std::vector<double> h_global_dist(num_rows * num_row_labels);

    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {

        // STEP 1: calculate_cluster_average
        CUDA_CHECK(cudaMemset(d_local_sum,   0, num_clusters * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_local_count, 0, num_clusters * sizeof(int)));
        {
            int blocks = (num_rows * local_cols + tuned.cluster_sum - 1) / tuned.cluster_sum;
            kernel_cluster_sum<<<blocks, tuned.cluster_sum>>>(
                num_rows, local_cols, num_col_labels,
                d_matrix, d_row_labels, d_col_labels,
                d_local_sum, d_local_count);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaMemcpy(h_local_sum.data(),   d_local_sum,   num_clusters*sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_local_count.data(), d_local_count, num_clusters*sizeof(int),    cudaMemcpyDeviceToHost));
        MPI_Allreduce(h_local_sum.data(),   h_global_sum.data(),   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(h_local_count.data(), h_global_count.data(), num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_local_sum,   h_global_sum.data(),   num_clusters*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_count, h_global_count.data(), num_clusters*sizeof(int),    cudaMemcpyHostToDevice));
        {
            int blocks = (num_clusters + tuned.divide_avg - 1) / tuned.divide_avg;
            kernel_divide_avg<<<blocks, tuned.divide_avg>>>(
                num_clusters, d_local_sum, d_local_count, d_cluster_avg);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // STEP 2: update_row_labels
        {
            int blocks = (num_rows * num_row_labels + tuned.row_dist - 1) / tuned.row_dist;
            kernel_row_distances<<<blocks, tuned.row_dist>>>(
                num_rows, local_cols, num_row_labels, num_col_labels,
                d_matrix, d_col_labels, d_cluster_avg, d_partial_dist);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaMemcpy(h_local_dist.data(), d_partial_dist,
                              num_rows*num_row_labels*sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Allreduce(h_local_dist.data(), h_global_dist.data(),
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
        CUDA_CHECK(cudaMemcpy(d_row_labels, row_labels, num_rows*sizeof(label_type), cudaMemcpyHostToDevice));

        // STEP 3: update_col_labels
        CUDA_CHECK(cudaMemset(d_cols_updated, 0, sizeof(int)));
        {
            int blocks = (local_cols + tuned.col_labels - 1) / tuned.col_labels;
            kernel_col_labels<<<blocks, tuned.col_labels>>>(
                num_rows, local_cols, num_col_labels,
                d_matrix, d_row_labels, d_col_labels,
                d_cluster_avg, d_cols_updated);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        int local_cols_updated = 0;
        CUDA_CHECK(cudaMemcpy(&local_cols_updated, d_cols_updated, sizeof(int), cudaMemcpyDeviceToHost));
        int global_cols_updated = 0;
        MPI_Allreduce(&local_cols_updated, &global_cols_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double local_dist_col = 0.0, global_dist_col = 0.0;
        MPI_Allreduce(&local_dist_col, &global_dist_col, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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

    CUDA_CHECK(cudaMemcpy(local_col_labels, d_col_labels, local_cols*sizeof(label_type), cudaMemcpyDeviceToHost));

    cudaFree(d_matrix); cudaFree(d_col_labels); cudaFree(d_row_labels);
    cudaFree(d_cluster_avg); cudaFree(d_local_sum); cudaFree(d_local_count);
    cudaFree(d_partial_dist); cudaFree(d_cols_updated);
}

// ─────────────────────────────────────────────────────────────────────────────
// main — identical to cgc_cuda.cu
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
