"""2-D and PCA views of an embedding. UMAP takes a while on CPU, so the result is cached next to the embeddings."""

import numpy as np

from store import dataset_dir


def umap_path(name: str, model: str, n_neighbors: int, min_dist: float, seed: int):
    return dataset_dir(name) / f"umap-{model}-nn{n_neighbors}-d{min_dist}-s{seed}.npy"


def umap_2d(
    name: str, model: str, X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = 0,
    overwrite: bool = False,
) -> np.ndarray:
    path = umap_path(name, model, n_neighbors, min_dist, seed)
    if path.exists() and not overwrite:
        return np.load(path)
    xy = umap_fit(X, n_neighbors, min_dist, seed)
    np.save(path, xy)
    return xy


def umap_fit(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, seed: int = 0) -> np.ndarray:
    """A cosine UMAP of `X`, not cached (for a selection)."""
    import umap

    n_neighbors = max(2, min(n_neighbors, len(X) - 1))
    return umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed, compatibility_layout=True
    ).fit_transform(X).astype(np.float32)


def pca(X: np.ndarray, k: int = 8, seed: int = 0) -> np.ndarray:
    from sklearn.decomposition import PCA

    return PCA(n_components=min(k, X.shape[1]), random_state=seed).fit_transform(X)
