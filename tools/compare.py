import argparse
import numpy as np
from common import load_labels


def parse_arguments():
    parser = argparse.ArgumentParser(prog="compare two coclustering")
    parser.add_argument(
        "output", help="The output labels generated your implementation"
    )
    parser.add_argument(
        "reference",
        help="The reference labels that the output will be compared against",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("reading input files...")
    a_row_labels, a_col_labels = load_labels(args.output)
    b_row_labels, b_col_labels = load_labels(args.reference)

    # Check if number of labels match
    if (len(a_row_labels), len(a_col_labels)) != (
        len(b_row_labels),
        len(b_col_labels),
    ):
        print("error: input files have different number of labels")
        return

    num_row_labels = max(np.amax(a_row_labels), np.amax(b_row_labels)) + 1
    num_col_labels = max(np.amax(a_col_labels), np.amax(b_col_labels)) + 1

    # Count the independent clusters in A and B
    print("calculating NMI...")
    n = len(a_row_labels) * len(a_col_labels)

    a_sizes = np.outer(
        np.bincount(a_row_labels, minlength=num_row_labels),
        np.bincount(a_col_labels, minlength=num_col_labels),
    )

    b_sizes = np.outer(
        np.bincount(b_row_labels, minlength=num_row_labels),
        np.bincount(b_col_labels, minlength=num_col_labels),
    )

    # Count the joint clusters in A and B
    ab_sizes = np.outer(
        np.bincount(
            a_row_labels * num_row_labels + b_row_labels, minlength=num_row_labels**2
        ),
        np.bincount(
            a_col_labels * num_col_labels + b_col_labels, minlength=num_col_labels**2
        ),
    )

    a_sizes = a_sizes.reshape(num_row_labels, 1, num_col_labels, 1)
    b_sizes = b_sizes.reshape(1, num_row_labels, 1, num_col_labels)
    ab_sizes = ab_sizes.reshape(
        num_row_labels, num_row_labels, num_col_labels, num_col_labels
    )

    with np.errstate(all="ignore"):
        mutual = np.nansum(ab_sizes * np.log2((n * ab_sizes) / (a_sizes * b_sizes)))
        entropy_a = np.nansum((a_sizes * np.log2(a_sizes / n)))
        entropy_b = np.nansum((b_sizes * np.log2(b_sizes / n)))

    nmi = mutual / np.sqrt(entropy_a * entropy_b)
    print(f"normalized-mutual information: {nmi:.5f}")


if __name__ == "__main__":
    main()
