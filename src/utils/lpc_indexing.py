import numpy as np
from scipy.spatial.distance import pdist, squareform


def find_center_point(coords, k=4):
    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)

    k_smallest = np.sort(distances, axis=1)[:, :k]
    mean_knn = np.mean(k_smallest, axis=1)

    center_idx = np.argmin(mean_knn)
    return coords[center_idx], int(center_idx)


def cluster_axis_values(values, tol=30.0):
    """
    Cluster 1D-Werte entlang einer Achse.

    Werte, die näher als tol beieinander liegen, werden demselben
    Cluster zugeordnet. Rückgabe sind die Clusterzentren.
    """
    values = np.sort(np.asarray(values, dtype=float))

    if len(values) == 0:
        return np.array([], dtype=float)

    clusters = [[values[0]]]

    for v in values[1:]:
        current_center = np.mean(clusters[-1])
        if abs(v - current_center) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    centers = np.array([np.mean(c) for c in clusters], dtype=float)
    return centers


def get_centered_indices_for_cluster_count(n_clusters):
    """
    Erzeugt zentrierte Indizes für Cluster.

    Beispiele:
        4 -> [-2, -1, 1, 2]
        5 -> [-2, -1, 0, 1, 2]
    """
    if n_clusters <= 0:
        raise ValueError("n_clusters muss > 0 sein.")

    if n_clusters % 2 == 1:
        half = n_clusters // 2
        return np.arange(-half, half + 1, dtype=int)

    half = n_clusters // 2
    return np.array(list(range(-half, 0)) + list(range(1, half + 1)), dtype=int)


def assign_axis_indices(values, cluster_centers):
    """
    Ordnet jedem Wert den Index des nächstgelegenen Clusterzentrums zu.
    """
    values = np.asarray(values, dtype=float)
    cluster_centers = np.asarray(cluster_centers, dtype=float)

    distances = np.abs(values[:, None] - cluster_centers[None, :])
    nearest = np.argmin(distances, axis=1)

    centered_indices = get_centered_indices_for_cluster_count(len(cluster_centers))
    return centered_indices[nearest]


def sort_result_by_indices(result):
    """
    Sortiert ein Array der Form [idx_x, idx_y, x, y]
    zuerst nach idx_x und dann nach idx_y.
    """
    return result[np.lexsort((result[:, 1], result[:, 0]))]


def check_unique_indices(indices):
    unique_indices = {tuple(idx) for idx in indices}
    return len(unique_indices) == len(indices)


def assign_doe_indices(coords, axis_tol=30.0):
    """
    Erwartet coords als Array der Form (N, 2) mit [x, y].

    Rückgabe:
        Array der Form (N, 4):
        [idx_x, idx_y, x, y]
    """
    coords = np.asarray(coords, dtype=float)

    center_point, center_idx = find_center_point(coords)

    # Mittelpunkt aus der Clusterbildung ausschließen,
    # damit nur echte Spalten/Zeilen geclustert werden
    coords_wo_center = np.delete(coords, center_idx, axis=0)

    x_clusters = cluster_axis_values(coords_wo_center[:, 0], tol=axis_tol)
    y_clusters = cluster_axis_values(coords_wo_center[:, 1], tol=axis_tol)

    idx_x = assign_axis_indices(coords[:, 0], x_clusters)
    idx_y = assign_axis_indices(coords[:, 1], y_clusters)

    idx_x[center_idx] = 0
    idx_y[center_idx] = 0

    print("\n--- LPC INDEXING DEBUG ---")
    print("center_idx:", center_idx)
    print("center_point:", np.round(center_point, 3))
    print("x_clusters:", np.round(x_clusters, 3))
    print("y_clusters:", np.round(y_clusters, 3))
    print("idx_x:", idx_x)
    print("idx_y:", idx_y)

    result = np.column_stack((idx_x, idx_y, coords))
    result = sort_result_by_indices(result)

    return result


def analyze_coordinates(coords):
    result = assign_doe_indices(coords)

    if not check_unique_indices(result[:, :2].astype(int)):
        print("Warning: Assigned indices are not unique!")

    return result