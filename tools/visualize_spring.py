import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys

from matplotlib.widgets import Slider
from matplotlib.colors import ListedColormap
from common import load_labels, process_clusters, create_colormap, unflatten

YEARS = list(range(1980, 2020))
MIN_LONGITUDE = -124.7086181640625
MAX_LONGITUDE = -66.98287963867188
MIN_LATITUDE = 25.130369186401367
MAX_LATITUDE = 49.20521926879883


def parse_arguments():
    parser = argparse.ArgumentParser(prog="visualize spring coclustering")
    parser.add_argument("data", help="The data spring_data_X.npy file")
    parser.add_argument("index", help="The index spring_index_X.npy file")
    parser.add_argument(
        "labels", help="The labels.txt file name produced by your application"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("reading input files...")
    data = np.load(args.data)
    coords = np.load(args.index)
    time_labels, loc_labels = load_labels(args.labels)

    ntime = len(time_labels)
    nloc = len(loc_labels)

    # Check shapes. If these are incorrect, you are loading the wrong files
    if data.shape != (ntime, nloc):
        print(
            f"error: invalid data file {args.data}: should have dimensions ({ntime}, {nloc})"
        )
        return

    if coords.shape != (2, nloc):
        print(
            f"error: invalid index file {args.index}: should have dimensions (2, {nloc}))"
        )
        return

    print("calculating cluster means...")
    nclusters, clusters, means = process_clusters(time_labels, loc_labels, data)

    data_range = (np.amin(data), np.amax(data))
    year = 0
    extent = (
        MIN_LONGITUDE,
        MAX_LONGITUDE,
        MIN_LATITUDE,
        MAX_LATITUDE,
    )

    print("plotting results...")

    # The original data
    plt.subplot(221)
    plt.title("Original data")
    data_im = plt.imshow(unflatten(data[year], coords), clim=data_range, extent=extent)
    plt.colorbar()

    # The clusters visualized using `nclusters` colors
    plt.subplot(222)
    n, m = len(np.unique(time_labels)), len(np.unique(loc_labels))
    plt.title(
        f"Clusters\n({n} temporal labels X {m} spatial labels = {nclusters} clusters)"
    )
    cmap = create_colormap(n, m)
    clusters_im = plt.imshow(
        unflatten(clusters[year], coords),
        cmap=cmap,
        clim=(0, nclusters - 1),
        extent=extent,
    )
    plt.colorbar()

    # Cluster averages
    plt.subplot(223)
    plt.title("Cluster average")
    means_im = plt.imshow(
        unflatten(means[year], coords), clim=data_range, extent=extent
    )
    plt.colorbar()

    # Difference beween original values and cluster averages
    plt.subplot(224)
    plt.title("Difference cluster average and original data")
    diff_max = np.amax(np.abs(means - data))
    diff_im = plt.imshow(
        np.abs(unflatten(means[year] - data[year], coords)),
        cmap="gnuplot",
        clim=(0, diff_max),
        extent=extent,
    )
    plt.colorbar()

    # Updates the images when selecting a different year
    def on_changed(value):
        year = YEARS.index(value)

        data_im.set_data(unflatten(data[year], coords))
        means_im.set_data(unflatten(means[year], coords))
        diff_im.set_data(np.abs(unflatten(means[year] - data[year], coords)))
        clusters_im.set_data(unflatten(clusters[year], coords))

    fig = plt.gcf()
    ax = fig.add_axes([0.1, 0.05, 0.8, 0.03])
    slider = Slider(ax, "Year", min(YEARS), max(YEARS), valstep=1)
    slider.on_changed(on_changed)

    plt.show()


if __name__ == "__main__":
    main()
