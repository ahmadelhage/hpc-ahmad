#include <chrono>
#include <iostream>
#include <limits>
#include <mpi.h>
#include "common.h"

std::vector<double> calculate_cluster_average(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    const label_type* local_col_labels)
{
    int num_clusters = num_row_labels * num_col_labels;

    std::vector<double> local_sum(num_clusters, 0.0);
    std::vector<int>    local_count(num_clusters, 0);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < local_cols; j++) {

            int r = row_labels[i];
            int c = local_col_labels[j];
            int idx = r * num_col_labels + c;

            local_sum[idx] += local_matrix[i * local_cols + j];
            local_count[idx] += 1;
        }
    }

    std::vector<double> global_sum(num_clusters, 0.0);
    std::vector<int> global_count(num_clusters, 0);

    MPI_Allreduce(local_sum.data(), global_sum.data(),
                  num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(local_count.data(), global_count.data(),
                  num_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_clusters, 0.0);

    for (int i = 0; i < num_clusters; i++) {
        if (global_count[i] > 0) {
            cluster_avg[i] = global_sum[i] / (double)global_count[i];
        } else {
            cluster_avg[i] = 0.0; // IMPORTANT: no "prev avg" hack
        }
    }

    return cluster_avg;
}

double calculate_distance(double avg, double item) {
    double d = avg - item;
    return d * d;
}

std::pair<int, double> update_row_labels(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    const label_type* local_col_labels,
    const double* cluster_avg)
{
    std::vector<double> local_dist(num_rows * num_row_labels, 0.0);

    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {

            double dist = 0.0;

            for (int j = 0; j < local_cols; j++) {
                int c = local_col_labels[j];
                int idx = k * num_col_labels + c;

                dist += calculate_distance(
                    cluster_avg[idx],
                    local_matrix[i * local_cols + j]
                );
            }

            local_dist[i * num_row_labels + k] = dist;
        }
    }

    std::vector<double> global_dist(num_rows * num_row_labels, 0.0);

    MPI_Allreduce(local_dist.data(), global_dist.data(),
                  num_rows * num_row_labels, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int changed = 0;
    double total = 0.0;

    for (int i = 0; i < num_rows; i++) {

        int best = 0;
        double best_d = global_dist[i * num_row_labels];

        for (int k = 1; k < num_row_labels; k++) {
            double d = global_dist[i * num_row_labels + k];
            if (d < best_d) {
                best_d = d;
                best = k;
            }
        }

        if (row_labels[i] != best) {
            row_labels[i] = best;
            changed++;
        }

        total += best_d;
    }

    return {changed, total};
}

std::pair<int, double> update_col_labels(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* cluster_avg)
{
    int changed = 0;
    double total = 0.0;

    for (int j = 0; j < local_cols; j++) {

        int best = 0;
        double best_d = std::numeric_limits<double>::infinity();

        for (int k = 0; k < num_col_labels; k++) {

            double dist = 0.0;

            for (int i = 0; i < num_rows; i++) {
                int r = row_labels[i];
                int idx = r * num_col_labels + k;

                dist += calculate_distance(
                    cluster_avg[idx],
                    local_matrix[i * local_cols + j]
                );
            }

            if (dist < best_d) {
                best_d = dist;
                best = k;
            }
        }

        if (local_col_labels[j] != best) {
            local_col_labels[j] = best;
            changed++;
        }

        total += best_d;
    }

    return {changed, total};
}

void cluster_mpi(
    int rank,
    int nprocs,
    int num_rows,
    int num_cols,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    label_type* local_col_labels,
    int max_iterations)
{
    int iter = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (iter < max_iterations) {

        auto cluster_avg = calculate_cluster_average(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels
        );

        auto [row_changed, dist_r] = update_row_labels(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            cluster_avg.data()
        );

        auto [col_changed, dist_c] = update_col_labels(
            num_rows, local_cols,
            num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            cluster_avg.data()
        );

        int global_changed = 0;
        MPI_Allreduce(&col_changed, &global_changed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double global_dist_c = 0.0;
        MPI_Allreduce(&dist_c, &global_dist_c, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        double total_dist = dist_r + global_dist_c;

        iter++;

        int total_changes = row_changed + global_changed;

        if (rank == 0) {
            std::cout << "iter " << iter
                      << " changes=" << total_changes
                      << " dist=" << total_dist << "\n";
        }

        if (total_changes == 0) break;
    }

    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        std::cout << "time: "
                  << std::chrono::duration<double>(end - start).count()
                  << " sec\n";
    }
}