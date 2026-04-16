#include <chrono>
#include <iostream>
#include <vector>
#include <limits>
#include <mpi.h>

#include "common.h"

/* ---------------- CLUSTER AVERAGE ---------------- */

std::vector<double> calculate_cluster_average(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    const label_type* local_col_labels) {

    int num_clusters = num_row_labels * num_col_labels;

    std::vector<double> local_sum(num_clusters, 0.0);
    std::vector<int>    local_count(num_clusters, 0);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < local_cols; j++) {

            double item = (double)local_matrix[i * local_cols + j];

            int r = row_labels[i];
            int c = local_col_labels[j];

            int idx = r * num_col_labels + c;

            local_sum[idx] += item;
            local_count[idx] += 1;
        }
    }

    std::vector<double> global_sum(num_clusters, 0.0);
    std::vector<int>    global_count(num_clusters, 0);

    MPI_Allreduce(local_sum.data(), global_sum.data(),
                  num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(local_count.data(), global_count.data(),
                  num_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> cluster_avg(num_clusters, 0.0);

    for (int i = 0; i < num_clusters; i++) {
        if (global_count[i] > 0) {
            cluster_avg[i] = global_sum[i] / (double)global_count[i];
        } else {
            cluster_avg[i] = 0.0;   // MUST match reference behavior
        }
    }

    return cluster_avg;
}

/* ---------------- DISTANCE ---------------- */

double calculate_distance(double avg, double item) {
    double d = avg - item;
    return d * d;
}

/* ---------------- ROW UPDATE ---------------- */

std::pair<int, double> update_row_labels(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    const label_type* local_col_labels,
    const double* cluster_avg) {

    std::vector<double> local_dist(num_rows * num_row_labels, 0.0);

    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {

            double dist = 0.0;

            for (int j = 0; j < local_cols; j++) {
                double item = local_matrix[i * local_cols + j];
                int cl = k * num_col_labels + local_col_labels[j];

                dist += calculate_distance(cluster_avg[cl], item);
            }

            local_dist[i * num_row_labels + k] = dist;
        }
    }

    std::vector<double> global_dist(num_rows * num_row_labels, 0.0);

    MPI_Allreduce(local_dist.data(), global_dist.data(),
                  num_rows * num_row_labels,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int num_updated = 0;
    double total_dist = 0.0;

    for (int i = 0; i < num_rows; i++) {

        int best_label = 0;
        double best_dist = global_dist[i * num_row_labels];

        for (int k = 1; k < num_row_labels; k++) {
            double d = global_dist[i * num_row_labels + k];

            if (d < best_dist) {
                best_dist = d;
                best_label = k;
            }
        }

        if (row_labels[i] != best_label) {
            row_labels[i] = best_label;
            num_updated++;
        }

        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}

/* ---------------- COL UPDATE ---------------- */

std::pair<int, double> update_col_labels(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* cluster_avg) {

    int num_updated = 0;
    double local_dist_sum = 0.0;

    for (int j = 0; j < local_cols; j++) {

        int best_label = 0;
        double best_dist = std::numeric_limits<double>::infinity();

        for (int k = 0; k < num_col_labels; k++) {

            double dist = 0.0;

            for (int i = 0; i < num_rows; i++) {
                double item = local_matrix[i * local_cols + j];
                int cl = row_labels[i] * num_col_labels + k;

                dist += calculate_distance(cluster_avg[cl], item);
            }

            if (dist < best_dist) {
                best_dist = dist;
                best_label = k;
            }
        }

        if (local_col_labels[j] != best_label) {
            local_col_labels[j] = best_label;
            num_updated++;
        }

        local_dist_sum += best_dist;
    }

    return {num_updated, local_dist_sum};
}

/* ---------------- MPI ITERATION ---------------- */

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
    int max_iterations) {

    int iteration = 0;

    auto start = std::chrono::high_resolution_clock::now();

    while (iteration < max_iterations) {

        auto cluster_avg = calculate_cluster_average(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix, row_labels, local_col_labels);

        auto [rows_updated, dist_row] = update_row_labels(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            cluster_avg.data());

        auto [cols_updated, dist_col] = update_col_labels(
            num_rows, local_cols,
            num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            cluster_avg.data());

        int global_cols_updated = 0;
        MPI_Allreduce(&cols_updated, &global_cols_updated,
                      1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double local_total = dist_row + dist_col;
        double global_total = 0.0;

        MPI_Allreduce(&local_total, &global_total,
                      1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        int num_updated = rows_updated + global_cols_updated;

        if (rank == 0) {
            std::cout << "iter " << iteration
                      << " updates=" << num_updated
                      << " dist=" << global_total
                      << "\n";
        }

        if (num_updated == 0) break;

        iteration++;
    }

    if (rank == 0) {
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "time: "
                  << std::chrono::duration<double>(end - start).count()
                  << "s\n";
    }
}