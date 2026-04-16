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

    int K = num_row_labels * num_col_labels;

    std::vector<double> sum(K, 0.0);
    std::vector<int> count(K, 0);

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < local_cols; j++) {

            int r = row_labels[i];
            int c = local_col_labels[j];

            int idx = r * num_col_labels + c;

            sum[idx] += local_matrix[i * local_cols + j];
            count[idx] += 1;
        }
    }

    std::vector<double> gsum(K, 0.0);
    std::vector<int> gcount(K, 0);

    MPI_Allreduce(sum.data(), gsum.data(), K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(count.data(), gcount.data(), K, MPI_SUM, MPI_COMM_WORLD);

    std::vector<double> avg(K, 0.0);

    for (int i = 0; i < K; i++) {
        avg[i] = (gcount[i] > 0) ? gsum[i] / gcount[i] : 0.0;
    }

    return avg;
}

/* ---------------- DIST ---------------- */

static inline double sq(double x) {
    return x * x;
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

    std::vector<double> dist(num_rows * num_row_labels, 0.0);

    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {

            double s = 0.0;

            for (int j = 0; j < local_cols; j++) {

                double v = local_matrix[i * local_cols + j];

                int idx = k * num_col_labels + local_col_labels[j];

                s += sq(avg[idx] - v);
            }

            dist[i * num_row_labels + k] = s;
        }
    }

    std::vector<double> gdist(num_rows * num_row_labels, 0.0);

    MPI_Allreduce(dist.data(), gdist.data(),
                  num_rows * num_row_labels,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int changed = 0;
    double total = 0.0;

    for (int i = 0; i < num_rows; i++) {

        int best = 0;
        double bestv = gdist[i * num_row_labels];

        for (int k = 1; k < num_row_labels; k++) {
            double v = gdist[i * num_row_labels + k];
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

            double s = 0.0;

            for (int i = 0; i < num_rows; i++) {

                double v = local_matrix[i * local_cols + j];

                int idx = row_labels[i] * num_col_labels + k;

                s += sq(avg[idx] - v);
            }

            if (s < bestv) {
                bestv = s;
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

/* ---------------- DRIVER ---------------- */

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
            local_matrix,
            row_labels,
            local_col_labels);

        auto [rchg, rdist] = update_row_labels(
            num_rows, local_cols,
            num_row_labels, num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            avg.data());

        auto [cchg, cdist] = update_col_labels(
            num_rows, local_cols,
            num_col_labels,
            local_matrix,
            row_labels,
            local_col_labels,
            avg.data());

        int global_c = 0;
        MPI_Allreduce(&cchg, &global_c, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        double local_total = rdist + cdist;
        double global_total = 0.0;

        MPI_Allreduce(&local_total, &global_total,
                      1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        int changed = rchg + global_c;

        if (rank == 0) {
            std::cout << "iter " << it
                      << " changed=" << changed
                      << " dist=" << global_total << "\n";
        }

        if (changed == 0) break;

        it++;
    }
}