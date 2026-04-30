#include <chrono>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <cuda_runtime.h>

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

// KERNEL 1: accumulate partial cluster sums and counts
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
        dist += diff * diff;
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
    int num_clusters = num_row_labels * num_col_labels;
    const int THREADS = 256;

    float*      d_matrix;
    label_type* d_col_labels;
    label_type* d_row_labels;
    double*     d_cluster_avg;
    double*     d_local_sum;
    int*        d_local_count;
    double*     d_partial_dist;
    int*        d_cols_updated;

    CUDA_CHECK(cudaMalloc(&d_matrix,       num_rows * local_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_col_labels,   local_cols            * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&d_row_labels,   num_rows              * sizeof(label_type)));
    CUDA_CHECK(cudaMalloc(&d_cluster_avg,  num_clusters          * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_sum,    num_clusters          * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_count,  num_clusters          * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_dist, num_rows * num_row_labels * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cols_updated, sizeof(int)));

    // Upload matrix and initial labels to GPU
    CUDA_CHECK(cudaMemcpy(d_matrix,     local_matrix,      num_rows * local_cols * sizeof(float),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_labels, local_col_labels,  local_cols            * sizeof(label_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_row_labels, row_labels,        num_rows              * sizeof(label_type), cudaMemcpyHostToDevice));

    // Host buffers for MPI reductions
    std::vector<double> h_local_sum(num_clusters);
    std::vector<int>    h_local_count(num_clusters);
    std::vector<double> h_global_sum(num_clusters);
    std::vector<int>    h_global_count(num_clusters);
    std::vector<double> h_local_dist(num_rows * num_row_labels);
    std::vector<double> h_global_dist(num_rows * num_row_labels);

    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {

        // calculatecluster average
        CUDA_CHECK(cudaMemset(d_local_sum,   0, num_clusters * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_local_count, 0, num_clusters * sizeof(int)));

        {
            int blocks = (num_rows * local_cols + THREADS - 1) / THREADS;
            kernel_cluster_sum<<<blocks, THREADS>>>(
                num_rows, local_cols, num_col_labels,
                d_matrix, d_row_labels, d_col_labels,
                d_local_sum, d_local_count);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy partial sums from GPU to host for MPI reduction
        CUDA_CHECK(cudaMemcpy(h_local_sum.data(),   d_local_sum,   num_clusters * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_local_count.data(), d_local_count, num_clusters * sizeof(int),    cudaMemcpyDeviceToHost));

        MPI_Allreduce(h_local_sum.data(),   h_global_sum.data(),   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(h_local_count.data(), h_global_count.data(), num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        // Copy global sums back to GPU compute averages there
        CUDA_CHECK(cudaMemcpy(d_local_sum,   h_global_sum.data(),   num_clusters * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_count, h_global_count.data(), num_clusters * sizeof(int),    cudaMemcpyHostToDevice));

        {
            int blocks = (num_clusters + THREADS - 1) / THREADS;
            kernel_divide_avg<<<blocks, THREADS>>>(
                num_clusters, d_local_sum, d_local_count, d_cluster_avg);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        //update_row_labels
        {
            int blocks = (num_rows * num_row_labels + THREADS - 1) / THREADS;
            kernel_row_distances<<<blocks, THREADS>>>(
                num_rows, local_cols, num_row_labels, num_col_labels,
                d_matrix, d_col_labels, d_cluster_avg, d_partial_dist);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Copy partial distances to host, reduce across MPI ranks
        CUDA_CHECK(cudaMemcpy(h_local_dist.data(), d_partial_dist,
                              num_rows * num_row_labels * sizeof(double), cudaMemcpyDeviceToHost));

        MPI_Allreduce(h_local_dist.data(), h_global_dist.data(),
                      num_rows * num_row_labels, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Pick best row label on CPU (tiny: 40 x num_row_labels)
        int    rows_updated = 0;
        double total_dist_row = 0.0;
        for (int i = 0; i < num_rows; i++) {
            int    best = 0;
            double best_dist = h_global_dist[i * num_row_labels];
            for (int k = 1; k < num_row_labels; k++) {
                double d = h_global_dist[i * num_row_labels + k];
                if (d < best_dist) { best_dist = d; best = k; }
            }
            if (row_labels[i] != best) { row_labels[i] = best; rows_updated++; }
            total_dist_row += best_dist;
        }

        // Sync updated row_labels back to GPU
        CUDA_CHECK(cudaMemcpy(d_row_labels, row_labels, num_rows * sizeof(label_type), cudaMemcpyHostToDevice));

        // STEP 3: update_col_labels
        CUDA_CHECK(cudaMemset(d_cols_updated, 0, sizeof(int)));

        {
            int blocks = (local_cols + THREADS - 1) / THREADS;
            kernel_col_labels<<<blocks, THREADS>>>(
                num_rows, local_cols, num_col_labels,
                d_matrix, d_row_labels, d_col_labels,
                d_cluster_avg, d_cols_updated);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int local_cols_updated = 0;
        CUDA_CHECK(cudaMemcpy(&local_cols_updated, d_cols_updated, sizeof(int), cudaMemcpyDeviceToHost));

        // Sum changed col-label counts across all MPI ranks
        int global_cols_updated = 0;
        MPI_Allreduce(&local_cols_updated, &global_cols_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Also sum col distances for reporting
        double local_dist_col = 0.0;
        double global_dist_col = 0.0;
        MPI_Allreduce(&local_dist_col, &global_dist_col, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        iteration++;

        int    num_updated   = rows_updated + global_cols_updated;
        double average_dist  = total_dist_row / (double)(num_rows * num_cols);

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


    cudaFree(d_matrix);
    cudaFree(d_col_labels);
    cudaFree(d_row_labels);
    cudaFree(d_cluster_avg);
    cudaFree(d_local_sum);
    cudaFree(d_local_count);
    cudaFree(d_partial_dist);
    cudaFree(d_cols_updated);
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
