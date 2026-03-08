#pragma once

#include <algorithm>
#include <cstdlib>
#include <random>
#include <regex>
#include <unordered_set>

#include "argparse/argparse.hpp"
#include "npy.hpp"

using label_type = int;

static bool read_labels(
    const std::string& file_name,
    int num_rows,
    int num_cols,
    label_type* row_labels,
    label_type* col_labels) {
    auto in = std::ifstream {file_name};

    if (!in) {
        fprintf(stderr, "error: could not open: %s\n", file_name.c_str());
        return false;
    }

    for (int i = 0; i < num_rows; i++) {
        in >> row_labels[i];

        if (row_labels[i] < 0 || row_labels[i] >= num_rows) {
            fprintf(
                stderr,
                "error: label of row %d is invalid: %d\n",
                i + 1,
                row_labels[i]);
            return false;
        }
    }

    for (int i = 0; i < num_cols; i++) {
        in >> col_labels[i];

        if (col_labels[i] < 0 || col_labels[i] >= num_cols) {
            fprintf(
                stderr,
                "error: label of column %d is invalid: %d\n",
                i + 1,
                col_labels[i]);
            return false;
        }
    }

    if (in.eof()) {
        fprintf(
            stderr,
            "error: label file has insufficient labels for input data: %s\n",
            file_name.c_str());
        return false;
    }

    if (!in) {
        fprintf(
            stderr,
            "error: error occurred while reading file: %s\n",
            file_name.c_str());
        return false;
    }

    return true;
}

static void write_labels(
    const std::string& file_name,
    int num_rows,
    int num_cols,
    const label_type* row_labels,
    const label_type* col_labels) {
    fprintf(stderr, "writing result to %s\n", file_name.c_str());
    auto out = std::ofstream {file_name};

    for (int i = 0; i < num_rows; i++) {
        out << row_labels[i] << "\n";
    }

    out << "\n";

    for (int i = 0; i < num_cols; i++) {
        out << col_labels[i] << "\n";
    }

    out << "\n";
}

template<typename R>
static std::vector<label_type>
initialize_labels(int num_items, int num_labels, R& rng) {
    auto labels = std::vector<label_type>(num_items);

    for (int i = 0; i < num_items; i++) {
        labels[i] = i % num_labels;
    }

    std::shuffle(labels.begin(), labels.end(), rng);
    return labels;
}

static bool parse_arguments(
    int argc,
    const char* argv[],
    int* num_rows_out,
    int* num_cols_out,
    int* num_row_labels_out,
    int* num_col_labels_out,
    std::vector<float>* matrix_out,
    std::vector<label_type>* row_labels_out,
    std::vector<label_type>* col_labels_out,
    std::string* result_file_out,
    int* max_iter_out) {
    auto program = argparse::ArgumentParser(argv[0]);
    program.add_argument("input-data")
        .help("Path to input data file in NPY format");

    program.add_argument("input-labels")
        .help("Path to the file containing the initial labels");

    program.add_argument("--seed", "-s")
        .scan<'i', int>()
        .help("Random seed used for initialization")
        .default_value(1);

    program.add_argument("--output", "-o")
        .help("Path to output file")
        .default_value(std::string("labels.txt"));

    program.add_argument("--max-iterations", "-m")
        .scan<'i', int>()
        .help("Maximum number of iterations")
        .default_value(100);

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        fprintf(stderr, "error: %s\n", err.what());
        return false;
    }

    std::string input_file = program.get("input-data");
    std::vector<unsigned long> shape;
    std::vector<float> matrix;

    try {
        npy::LoadArrayFromNumpy(input_file, shape, matrix);
    } catch (const std::exception& e) {
        fprintf(
            stderr,
            "error while loading %s: %s\n",
            input_file.c_str(),
            e.what());
        return false;
    }

    if (shape.size() != 2) {
        fprintf(
            stderr,
            "input data must be two-dimensional: %s\n",
            input_file.c_str());
        return false;
    }

    int num_rows = int(shape[0]);
    int num_cols = int(shape[1]);

    std::vector<label_type> row_labels(num_rows);
    std::vector<label_type> col_labels(num_cols);

    int num_row_labels, num_col_labels;
    std::string input_labels = program.get("input-labels");
    std::smatch match;

    if (std::regex_match(
            input_labels,
            match,
            std::regex("([0-9]+)x([0-9]+)"))) {
        auto seed = program.get<int>("seed");
        auto rng = std::default_random_engine(seed);

        num_row_labels = std::stoi(match[1]);
        num_col_labels = std::stoi(match[2]);

        row_labels = initialize_labels(num_rows, num_row_labels, rng);
        col_labels = initialize_labels(num_cols, num_col_labels, rng);
    } else {
        if (!read_labels(
                input_labels,
                num_rows,
                num_cols,
                row_labels.data(),
                col_labels.data())) {
            return false;
        }

        num_row_labels =
            *std::max_element(row_labels.begin(), row_labels.end()) + 1;
        num_col_labels =
            *std::max_element(col_labels.begin(), col_labels.end()) + 1;
    }

    auto max_iter = program.get<int>("max-iterations");
    auto file_out = program.get("output");

    fprintf(stderr, "arguments:\n");
    fprintf(
        stderr,
        " * matrix %s: %d x %d\n",
        input_file.c_str(),
        num_rows,
        num_cols);
    fprintf(stderr, " * row labels: %d\n", num_row_labels);
    fprintf(stderr, " * column labels: %d\n", num_col_labels);
    fprintf(stderr, " * output: %s\n", file_out.c_str());
    fprintf(stderr, " * max. iterations: %d\n", max_iter);

    *num_rows_out = num_rows;
    *num_cols_out = num_cols;
    *num_row_labels_out = num_row_labels;
    *num_col_labels_out = num_col_labels;
    *row_labels_out = row_labels;
    *col_labels_out = col_labels;
    *matrix_out = std::move(matrix);
    *result_file_out = file_out;
    *max_iter_out = max_iter;
    return true;
}
