#include <chrono>
#include <iostream>
#include <mpi.h>

#include "common.h"

/**
 * This function returns a matrix of size (num_row_labels, num_col_labels)
 * that stores the average value for each combination of row label and
 * column label. In other words, the entry at coordinate (x, y) is the
 * average over all input values having row label x and column label y.
 */
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

    // Each process calculates the sum and count for its local portion of the matrix
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            float item      = local_matrix[i * local_cols + j];
            int   row_label = row_labels[i];
            int   col_label = local_col_labels[j];
            int   idx       = row_label * num_col_labels + col_label;
            local_sum[idx]   += item;
            local_count[idx] += 1;
        }
    }

    std::vector<double> global_sum(num_clusters, 0.0);
    std::vector<int>    global_count(num_clusters, 0);
    // Use MPI_Allreduce to sum up the local sums and counts across all processes
    MPI_Allreduce(local_sum.data(),   global_sum.data(),   num_clusters, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(local_count.data(), global_count.data(), num_clusters, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);
    
    // Calculate the average for each cluster using the global sums and counts
    std::vector<double> cluster_avg(num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        cluster_avg[i] = (global_count[i] > 0)
            ? global_sum[i] / double(global_count[i])
            : 0.0;
    }
    return cluster_avg;
}

double calculate_distance(double avg, double item) {
    double diff = (avg - item);
    return diff * diff;
}

/**
 * Update row labels. Each process computes partial distances (summed over
 * its local columns) for all rows x all candidate row-labels.
 * MPI_Allreduce sums these partial distances so every process has the full
 * distance and can independently pick the best label for each row.
 *
 * Returns {num_rows_changed, total_dist}.
 */
std::pair<int, double> update_row_labels(
    int num_rows,
    int local_cols,
    int num_row_labels,
    int num_col_labels,
    const float* local_matrix,
    label_type* row_labels,
    const label_type* local_col_labels,
    const double* cluster_avg) {

    // local_dist[i * num_row_labels + k] = partial distance for row i, candidate label k
    std::vector<double> local_dist(num_rows * num_row_labels, 0.0);

    // Each process computes the partial distance for its local columns
    for (int i = 0; i < num_rows; i++) {
        for (int k = 0; k < num_row_labels; k++) {
            double dist = 0.0;
            for (int j = 0; j < local_cols; j++) {
                // Calculate the distance between the item and the cluster average for the candidate label
                double item = local_matrix[i * local_cols + j];
                int    cl   = k * num_col_labels + local_col_labels[j];
                dist += calculate_distance(cluster_avg[cl], item);
            }
            local_dist[i * num_row_labels + k] = dist;
        }
    }

    /* Use MPI_Allreduce to sum up the local distances across all processes, so each process has the 
    total distance for each row and candidate label */
    std::vector<double> global_dist(num_rows * num_row_labels, 0.0);
    MPI_Allreduce(local_dist.data(), global_dist.data(),
                  num_rows * num_row_labels, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    int    num_updated = 0;
    double total_dist  = 0.0;

    // Each process independently picks the best label for each row based on the global distances
    for (int i = 0; i < num_rows; i++) {
        int    best_label = 0;
        double best_dist  = global_dist[i * num_row_labels];

        // Iterate over candidate labels to find the one with the smallest distance
        for (int k = 1; k < num_row_labels; k++) {
            double d = global_dist[i * num_row_labels + k];
            if (d < best_dist) {
                best_dist  = d;
                best_label = k;
            }
        }
        // Update the row label if it has changed and count the number of updates
        if (row_labels[i] != best_label) {
            row_labels[i] = best_label;
            num_updated++;
        }
        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}

/**
 * Update the labels aong the columns. Each process computes the distance for its local 
 * columns and updates the labels independently, without needing to communicate with other 
 * processes, since the row labels and cluster averages are already synchronized across all 
 * processes.
 */

std::pair<int, double> update_col_labels(
    int num_rows,
    int local_cols,
    int num_col_labels,
    const float* local_matrix,
    const label_type* row_labels,
    label_type* local_col_labels,
    const double* cluster_avg) {

    int    num_updated = 0;
    double total_dist  = 0.0;

    for (int j = 0; j < local_cols; j++) {
        int    best_label = 0;
        double best_dist  = std::numeric_limits<double>::infinity();

        for (int k = 0; k < num_col_labels; k++) {
            double dist = 0.0;
            for (int i = 0; i < num_rows; i++) {
                double item = local_matrix[i * local_cols + j];
                int    cl   = row_labels[i] * num_col_labels + k;
                dist += calculate_distance(cluster_avg[cl], item);
            }
            if (dist < best_dist) {
                best_dist  = dist;
                best_label = k;
            }
        }

        if (local_col_labels[j] != best_label) {
            local_col_labels[j] = best_label;
            num_updated++;
        }
        total_dist += best_dist;
    }

    return {num_updated, total_dist};
}
/**
 * Perform one iteration of the co-clustering algorithm. This function updates
 * the labels in both `row_labels` and `col_labels`, and returns the total
 * number of labels that changed (i.e., the number of rows and columns that
 * were reassigned to a different label).
 */
std::pair<int, double> cluster_serial_iteration(
    int num_rows,
    int num_cols,
    int num_row_labels,
    int num_col_labels,
    const float* matrix,
    label_type* row_labels,
    label_type* col_labels) {
    // Calculate the average value per cluster
    auto cluster_avg = calculate_cluster_average(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix,
        row_labels,
        col_labels);

    // Update labels along the rows
    auto [num_rows_updated, total_dist_row] = update_row_labels(
        num_rows,
        num_cols,
        num_row_labels,
        num_col_labels,
        matrix,
        row_labels,
        col_labels,
        cluster_avg.data());

    // Update the labels along the columns
    auto [num_cols_updated, total_dist_col] = update_col_labels(
        num_rows,
        num_cols,
        num_col_labels,
        matrix,
        row_labels,
        col_labels,
        cluster_avg.data());

    return {num_rows_updated + num_cols_updated, total_dist_row + total_dist_col};
}

/**
 * Columns are distributed across MPI processes.Row labels are replicated on all 
 * processes (40 rows). The global average for each cluster is calculated using MPI_Allreduce,
 *  which sums the local averages from each process and divides by the total count to get 
 * the global average. Row labels are updated using MPI_Allreduce to sum the local distances 
 * across all processes, allowing each process to independently determine the best label for 
 * each row. Column labels are updated independently on each process without communication, 
 * since they only depend on the row labels and cluster averages, which are already 
 * synchronized across all processes.
 */
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

        // Sum col label changes across all processes
        int global_cols_updated = 0;
        MPI_Allreduce(&local_cols_updated, &global_cols_updated, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Sum distances across all processes for reporting
        double total_dist = dist_row;
        double global_dist_col = 0.0;
        MPI_Allreduce(&dist_col, &global_dist_col, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        total_dist += global_dist_col;

        iteration++;

        int num_updated = rows_updated + global_cols_updated;
        double average_dist = total_dist / double(num_rows * num_cols);

        if (rank == 0) {
            std::cout << "iteration " << iteration << ": " << num_updated
                      << " labels were updated, average error is " << average_dist
                      << "\n";
        }

        if (num_updated == 0) {
            break;
        }
    }

    auto after = std::chrono::high_resolution_clock::now();
    auto time_seconds = std::chrono::duration<double>(after - before).count();

    if (rank == 0) {
        std::cout << "clustering time total: " << time_seconds << " seconds\n";
        std::cout << "clustering time per iteration: " << (time_seconds / iteration)
                  << " seconds\n";
    }
}


int main(int argc, const char* argv[]) {

    //initialize MPI
    MPI_Init(nullptr, nullptr);

    
    int rank, nprocs;
    //get rank and number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Start wall clock timer
    auto wall_start = std::chrono::high_resolution_clock::now();



    std::string output_file;
    std::vector<float> matrix;
    std::vector<label_type> row_labels, col_labels;
    int num_rows = 0, num_cols = 0;
    int num_row_labels = 0, num_col_labels = 0;
    int max_iter = 0;

    auto before = std::chrono::high_resolution_clock::now();

// Only the process with rank 0 will read the input and perform the clustering
    if(rank == 0) {    
        // Parse arguments
        if (!parse_arguments(
                argc,
                argv,
                &num_rows,
                &num_cols,
                &num_row_labels,
                &num_col_labels,
                &matrix,
                &row_labels,
                &col_labels,
                &output_file,
                &max_iter)) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
    }

    // Broadcast the input data to all processes
    MPI_Bcast(&num_rows,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_cols,       1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_row_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_col_labels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iter,       1, MPI_INT, 0, MPI_COMM_WORLD);

    //setting up the output file for all processes
    int fname_len = 0;
    if (rank == 0) fname_len = int(output_file.size());
    MPI_Bcast(&fname_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) output_file.resize(fname_len);
    MPI_Bcast(output_file.data(), fname_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    /*Calculate the number of columns each process will handle where ncolPerP represents the 
    number of columns for each process and disp is the index of the first column for each process */
    std::vector<int> ncolPerP(nprocs), disp(nprocs);
    int base = num_cols / nprocs;
    int rem  = num_cols % nprocs;
    for (int p = 0; p < nprocs; p++) {
        ncolPerP[p] = base + (p < rem ? 1 : 0);
        disp[p]     = (p == 0) ? 0 : disp[p - 1] + ncolPerP[p - 1];
    }
    int local_cols = ncolPerP[rank];

    std::vector<label_type> local_col_labels(local_cols);
    std::vector<float> local_matrix(num_rows * local_cols);

    /* Broadcast the row labels to all processes as they arent large and all processes 
    need them to perform the clustering */
    if (rank != 0) row_labels.resize(num_rows);
    MPI_Bcast(row_labels.data(), num_rows, MPI_INT, 0, MPI_COMM_WORLD);
    
    /*MPI_Scatterv is used to distribute the column labels to each process based 
    on the number of columns they will handle*/
    MPI_Scatterv(
    (rank == 0) ? col_labels.data() : nullptr,
    ncolPerP.data(), disp.data(), MPI_INT,
    local_col_labels.data(), local_cols, MPI_INT,
    0, MPI_COMM_WORLD);


    /* Each process scatters the corresponding columns of the matrix to all processes using 
    MPI_Scatterv. This allows each process to receive only the portion of the matrix that 
    it will work on, reducing memory usage and communication overhead. And since the number of 
    row is small, we can afford to send the whole row to each process without significant overhead.
    */
    for (int i = 0; i < num_rows; i++) {
        const float* row_src = (rank == 0) ? (matrix.data() + i * num_cols) : nullptr;
        float*       row_dst = local_matrix.data() + i * local_cols;
        MPI_Scatterv(row_src, ncolPerP.data(), disp.data(), MPI_FLOAT,
                    row_dst, local_cols, MPI_FLOAT,
                    0, MPI_COMM_WORLD);
    }
    
    // Clear the original matrix to free up memory as it is no longer needed after scattering
    if (rank == 0) { 
        matrix.clear(); 
        matrix.shrink_to_fit(); 
    }


    // Cluster labels
    cluster_mpi(
        rank, nprocs,
        num_rows, num_cols, local_cols,
        num_row_labels, num_col_labels,
        local_matrix.data(),
        row_labels.data(),
        local_col_labels.data(),
        max_iter);

    // Gather the updated column labels from all processes back to the root process (rank 0) using MPI_Gatherv
    MPI_Gatherv(
        local_col_labels.data(), local_cols, MPI_INT,
        (rank == 0) ? col_labels.data() : nullptr,
        ncolPerP.data(), disp.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    // Write resulting labels to output file (only by rank 0)
    if (rank == 0) {
        write_labels(
            output_file,
            num_rows, num_cols,
            row_labels.data(),
            col_labels.data());
    }

    // Finalize MPI and report total execution time
    auto wall_end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto time_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
        std::cout << "total execution time: " << time_seconds << " seconds\n";
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}
