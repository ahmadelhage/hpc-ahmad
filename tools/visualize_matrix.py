import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from common import load_labels, process_clusters, create_colormap


def parse_arguments():
    parser = argparse.ArgumentParser(prog="visualize spring coclustering")
    parser.add_argument("data", help="The input spring_data_X.npy file name")
    parser.add_argument(
        "labels", help="The labels.txt file produced by your application"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("reading input files...")
    data = np.load(args.data)
    time_labels, loc_labels = load_labels(args.labels)

    ntime = len(time_labels)
    nloc = len(loc_labels)

    # Check shapes. If these are incorrect, you are loading the wrong files
    assert data.shape == (ntime, nloc)

    print("calculating cluster means...")
    nclusters, clusters, means = process_clusters(time_labels, loc_labels, data)

    data_range = (np.amin(data), np.amax(data))

    if data.shape[1] > 1000 and False:
        factor = data.shape[1] // 1000
        data = data[:, ::factor]
        means = means[:, ::factor]
        clusters = clusters[:, ::factor]

    print("plotting results...")

    kwargs = {"aspect": "auto", "interpolation": "none"}

    # The original data
    plt.subplot(221)
    plt.title("Original data")
    data_im = plt.imshow(data, clim=data_range, **kwargs)
    plt.colorbar()

    # The clusters visualized using `nclusters` colors
    plt.subplot(222)
    n, m = len(np.unique(time_labels)), len(np.unique(loc_labels))
    plt.title(
        f"Clusters\n({n} temporal labels X {m} spatial labels = {nclusters} clusters)"
    )
    cmap = create_colormap(n, m)
    clusters_im = plt.imshow(
        clusters,
        cmap=cmap,
        clim=(0, nclusters - 1),
        **kwargs,
    )
    plt.colorbar()

    # Cluster averages
    plt.subplot(223)
    plt.title("Cluster average")
    means_im = plt.imshow(
        means,
        clim=data_range,
        **kwargs,
    )
    plt.colorbar()

    # Difference beween original values and cluster averages
    plt.subplot(224)
    plt.title("Difference cluster average and original data")
    diff_max = np.amax(np.abs(means - data))
    diff_im = plt.imshow(
        np.abs(means - data),
        cmap="gnuplot",
        clim=(0, diff_max),
        **kwargs,
    )
    plt.colorbar()

    plt.show()


if __name__ == "__main__":
    main()
