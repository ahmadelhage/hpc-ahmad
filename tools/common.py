import numpy as np


def load_labels(filename):
    # First line is time labels, second line is location labels
    with open(filename) as f:
        parts = f.read().split("\n\n", 2)
        time_labels = np.fromstring(parts[0], sep="\n", dtype=np.int32)
        loc_labels = np.fromstring(parts[1], sep="\n", dtype=np.int32)

    return time_labels, loc_labels


def process_clusters(time_labels, loc_labels, data):
    ntime = len(time_labels)
    nloc = len(loc_labels)
    clusters = np.zeros((ntime, nloc), dtype=np.int16)
    means = np.zeros((ntime, nloc))

    nclusters = 0
    for a in np.unique(time_labels):
        for b in np.unique(loc_labels):
            mask = (time_labels == a).reshape(-1, 1) & (loc_labels == b)
            clusters[mask] = nclusters
            means[mask] = np.average(data[mask])
            nclusters += 1

    return nclusters, clusters, means


def create_colormap(n, m):
    """Returns a matplotlib colormap consisting of `n*m` items: `n` different
    distinct colors with `m` different shades of each color.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = []
    tab10 = plt.get_cmap("tab10").colors

    for i in range(n):
        base = tab10[i % len(tab10)]

        for j in range(m):
            t = 0.9 * float(j) / m
            color = (1 - t) * np.array(base) + t
            colors.append(color)

    return ListedColormap(colors)


def unflatten(values, coords):
    """Transform 1D array into a 2D array where the XY coordinate of each value
    in `values` is given by `coords`
    """
    width = np.amax(coords[0]) + 1
    height = np.amax(coords[1]) + 1
    result = np.full((width, height), float("nan"))
    result[coords[0], coords[1]] = values
    return result
