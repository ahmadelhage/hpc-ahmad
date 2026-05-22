#include <chrono>
#include <iostream>
#include <limits>
#include <omp.h>
#include <mpi.h>

#include "common.h"

/**
 * calculate_cluster_average with OpenMP.
 * Each thread accumulates into its own private arrays to avoid race conditions.
 * Private arrays are combined at the end using a critical section.
 * Then MPI_Allreduce combines partial results across nodes.
 */
std::vector<double> calculate_cluster_average(
    int num_rows, int local_cols,
    int num_row_labels, int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    const label_type* local_col_labels)
{
    int num_clusters = num_row_labels * num_col_labels;
    std::vector<double> local_sum(num_clusters, 0.0);
    std::vector<int>    local_count(num_clusters, 0);

    #pragma omp parallel
    {
        // Private per-thread accumulators — no race conditions, no atomics
        std::vector<double> t_sum(num_clusters, 0.0);
        std::vector<int>    t_count(num_clusters, 0);

        #pragma omp for nowait schedule(static)
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                double item = (double)local_matrix[i * local_cols + j];
                int    idx  = row_labels[i] * num_col_labels + local_col_labels[j];
                t_sum[idx]   += item;
                t_count[idx] += 1;
            }
        }

        // Combine thread-private results into shared arrays
        #pragma omp critical
        {
            for (int i = 0; i < num_clusters; i++) {
                local_sum[i]   += t_sum[i];
                local_count[i] += t_count[i];
            }
        }
    }

    std::vector<double> global_sum(num_clusters);
    std::vector<int>    global_count(num_clusters);
    MPI_Allreduce(local_sum.data(),   global_sum.data(),   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_count.data(), global_count.data(), num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_clusters);
    for (int i = 0; i < num_clusters; i++)
        cluster_avg[i] = (global_count[i] > 0)
            ? global_sum[i] / (double)global_count[i]
            : 0.0;
    return cluster_avg;
}

/**
 * update_row_labels with OpenMP.
 * Each thread computes partial distances for its assigned rows independently.
 * MPI_Allreduce combines partial distances across nodes.
 * All processes pick the best label from identical global distances.
 */
std::pair<int, double> update_row_labels(
    int num_rows, int local_cols,
    int num_row_labels, int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    const label_type* local_col_labels,
    const double* cluster_avg)
{
    std::vector<double> local_dist(num_rows * num_row_labels, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0.0;
            for (int j = 0; j < local_cols; j++) {
                double item = (double)local_matrix[i * local_cols + j];
                int    cl   = k * num_col_labels + local_col_labels[j];
                double diff = cluster_avg[cl] - item;
                dist += diff * diff;
            }
            local_dist[i * num_row_labels + k] = dist;
        }
    }

    std::vector<double> global_dist(num_rows * num_row_labels);
    MPI_Allreduce(local_dist.data(), global_dist.data(),
                  num_rows * num_row_labels, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int    num_updated = 0;
    double total_dist  = 0.0;
    for (int i = 0; i < num_rows; i++) {
        int    best      = 0;
        double best_dist = global_dist[i * num_row_labels];
        for (int k = 1; k < num_row_labels; k++) {
            double d = global_dist[i * num_row_labels + k];
            if (d < best_dist) { best_dist = d; best = k; }
        }
        if (row_labels[i] != best) { row_labels[i] = best; num_updated++; }
        total_dist += best_dist;
    }
    return {num_updated, total_dist};
}

/**
 * update_col_labels with OpenMP.
 * Each thread handles a subset of local columns independently.
 * No MPI communication needed — cluster_avg and row_labels are already
 * globally consistent on every process.
 */
std::pair<int, double> update_col_labels(
    int num_rows, int local_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* cluster_avg)
{
    int    num_updated = 0;
    double total_dist  = 0.0;

    #pragma omp parallel for schedule(static) reduction(+:num_updated,total_dist)
    for (int j = 0; j < local_cols; j++) {
        int    best_label = 0;
        double best_dist  = std::numeric_limits<double>::infinity();

        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0.0;
            for (int i = 0; i < num_rows; i++) {
                double item = (double)local_matrix[i * local_cols + j];
                int    cl   = row_labels[i] * num_col_labels + k;
                double diff = cluster_avg[cl] - item;
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_label = k; }
        }

        if (local_col_labels[j] != best_label) {
            local_col_labels[j] = best_label;
            num_updated++;
        }
        total_dist += best_dist;
    }
    return {num_updated, total_dist};
}

void cluster_mpi_omp(
    int rank, int nprocs,
    int num_rows, int num_cols, int local_cols,
    int num_row_labels, int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    label_type* local_col_labels,
    int max_iterations)
{
    if (rank == 0)
        printf("[MPI+OpenMP] Using %d OpenMP threads per MPI process\n",
               omp_get_max_threads());

    int iteration = 0;
    auto before = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {

        auto cluster_avg = calculate_cluster_average(
            num_rows, local_cols, num_row_labels, num_col_labels,
            local_matrix, row_labels, local_col_labels);

        auto [rows_updated, dist_row] = update_row_labels(
            num_rows, local_cols, num_row_labels, num_col_labels,
            local_matrix, row_labels, local_col_labels, cluster_avg.data());

        auto [local_cols_updated, dist_col] = update_col_labels(
            num_rows, local_cols, num_col_labels,
            local_matrix, row_labels, local_col_labels, cluster_avg.data());

        int global_cols_updated = 0;
        MPI_Allreduce(&local_cols_updated, &global_cols_updated,
                      1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double global_dist_col = 0.0;
        MPI_Allreduce(&dist_col, &global_dist_col,
                      1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        iteration++;

        int    num_updated  = rows_updated + global_cols_updated;
        double average_dist = (dist_row + global_dist_col)
                              / (double)(num_rows * num_cols);

        if (rank == 0)
            std::cout << "iteration " << iteration << ": " << num_updated
                      << " labels were updated, average error is "
                      << average_dist << "\n";

        if (num_updated == 0) break;
    }

    auto after = std::chrono::high_resolution_clock::now();
    double time_seconds = std::chrono::duration<double>(after - before).count();
    if (rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: "
                  << (time_seconds / iteration) << " seconds\n";
    }
}

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

    cluster_mpi_omp(rank, nprocs, num_rows, num_cols, local_cols,
                    num_row_labels, num_col_labels,
                    local_matrix.data(), row_labels.data(),
                    local_col_labels.data(), max_iter);

    if (rank==0) col_labels.resize(num_cols);
    MPI_Gatherv(local_col_labels.data(), local_cols, MPI_INT,
                (rank==0)?col_labels.data():nullptr,
                sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank==0)
        write_labels(output_file, num_rows, num_cols,
                     row_labels.data(), col_labels.data());

    auto wall_end = std::chrono::high_resolution_clock::now();
    if (rank==0) {
        double t = std::chrono::duration<double>(wall_end-wall_start).count();
        std::cout << "total execution time: " << t << " seconds\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
