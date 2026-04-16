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

            double val = local_matrix[i * local_cols + j];

            int r = row_labels[i];
            int c = local_col_labels[j];

            int idx = r * num_col_labels + c;

            local_sum[idx] += val;
            local_count[idx] += 1;
        }
    }

    std::vector<double> global_sum(num_clusters, 0.0);
    std::vector<int>    global_count(num_clusters, 0);

    MPI_Allreduce(local_sum.data(), global_sum.data(),
                  num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(local_count.data(), global_count.data(),
                  num_clusters, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> avg(num_clusters, 0.0);

    for (int i = 0; i < num_clusters; i++) {
        if (global_count[i] > 0)
            avg[i] = global_sum[i] / (double)global_count[i];
        else
            avg[i] = 0.0;
    }

    return avg;
}

/* ---------------- DIST ---------------- */

double dist(double a, double b) {
    double d = a - b;
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
    const double* avg) {

    std::vector<double> local(num_rows * num_row_labels, 0.0);

    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {

            double sum = 0.0;

            for (int j = 0; j < local_cols; j++) {
                double v = local_matrix[i * local_cols + j];
                int idx = k * num_col_labels + local_col_labels[j];
                sum += dist(avg[idx], v);
            }

            local[i * num_row_labels + k] = sum;
        }
    }

    std::vector<double> global(num_rows * num_row_labels, 0.0);

    MPI_Allreduce(local.data(), global.data(),
                  num_rows * num_row_labels,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int changed = 0;
    double total = 0.0;

    for (int i = 0; i < num_rows; i++) {

        int best = 0;
        double bestv = global[i * num_row_labels];

        for (int k = 1; k < num_row_labels; k++) {
            double v = global[i * num_row_labels + k];
            if (v < bestv) {
                bestv = v;
                best = k;
            }
        }

        if (row_labels[i] != best) {
            row_labels[i] = best;
            changed++;
        }

        total += bestv;
    }

    /* 🔥 CRITICAL FIX: synchronize row labels immediately */
    MPI_Allreduce(MPI_IN_PLACE, row_labels,
                  num_rows, MPI_INT, MPI_BOR, MPI_COMM_WORLD);

    return {changed, total};
}

/* ---------------- COL UPDATE ---------------- */

std::pair<int, double> update_col_labels(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* avg) {

    int changed = 0;
    double total = 0.0;

    for (int j = 0; j < local_cols; j++) {

        int best = 0;
        double bestv = std::numeric_limits<double>::infinity();

        for (int k = 0; k < num_col_labels; k++) {

            double sum = 0.0;

            for (int i = 0; i < num_rows; i++) {
                double v = local_matrix[i * local_cols + j];
                int idx = row_labels[i] * num_col_labels + k;
                sum += dist(avg[idx], v);
            }

            if (sum < bestv) {
                bestv = sum;
                best = k;
            }
        }

        if (local_col_labels[j] != best) {
            local_col_labels[j] = best;
            changed++;
        }

        total += bestv;
    }

    return {changed, total};
}

/* ---------------- MPI DRIVER ---------------- */

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
    int max_iter) {

    int it = 0;

    while (it < max_iter) {

        auto avg = calculate_cluster_average(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix, row_labels, local_col_labels);

        auto [r_upd, d_row] = update_row_labels(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            avg.data());

        auto [c_upd, d_col] = update_col_labels(
            num_rows, local_cols,
            num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            avg.data());

        int global_c = 0;
        MPI_Allreduce(&c_upd, &global_c, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double local_total = d_row + d_col;
        double global_total = 0.0;

        MPI_Allreduce(&local_total, &global_total,
                      1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        int changed = r_upd + global_c;

        if (rank == 0) {
            std::cout << "iter " << it
                      << " changed=" << changed
                      << " dist=" << global_total
                      << "\n";
        }

        if (changed == 0) break;

        it++;
    }
}