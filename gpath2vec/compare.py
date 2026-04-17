"""cross-study comparison via canonical correlation analysis."""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA


def cca_compare(emb_a, emb_b, n_components=10):
    """
    canonical correlation analysis between two embedding matrices.

    emb_a, emb_b: numpy arrays (n_samples_a x dim), (n_samples_b x dim)
    n_components: number of canonical components

    returns: dict with correlations, transformed components, and summary
    """
    n = min(len(emb_a), len(emb_b))
    n_components = min(n_components, n, emb_a.shape[1])

    idx_a = np.random.choice(len(emb_a), n, replace=False)
    idx_b = np.random.choice(len(emb_b), n, replace=False)
    Xa = emb_a[idx_a]
    Xb = emb_b[idx_b]

    cca = CCA(n_components=n_components, max_iter=1000)
    Xa_c, Xb_c = cca.fit_transform(Xa, Xb)

    correlations = np.array([
        np.corrcoef(Xa_c[:, i], Xb_c[:, i])[0, 1]
        for i in range(n_components)
    ])

    return {
        "correlations": correlations,
        "mean_correlation": correlations.mean(),
        "components_a": Xa_c,
        "components_b": Xb_c,
        "n_samples": n,
        "n_components": n_components,
    }


def pairwise_cca(embeddings_dict, n_components=10):
    """
    run cca between all pairs of groups.

    embeddings_dict: {group_name: numpy array (n x dim)}
    returns: dataframe with group_a, group_b, mean_corr, per-component correlations
    """
    names = sorted(embeddings_dict.keys())
    results = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res = cca_compare(embeddings_dict[a], embeddings_dict[b],
                              n_components=n_components)
            row = {
                "group_a": a,
                "group_b": b,
                "n_samples": res["n_samples"],
                "mean_correlation": res["mean_correlation"],
            }
            for k, c in enumerate(res["correlations"]):
                row[f"cc_{k+1}"] = c
            results.append(row)

    return pd.DataFrame(results)


def cosine_similarity_matrix(embeddings_dict):
    """
    cosine similarity between group mean embeddings.

    embeddings_dict: {group_name: numpy array (n x dim)}
    returns: dataframe (n_groups x n_groups) of cosine similarities
    """
    from sklearn.metrics.pairwise import cosine_similarity

    names = sorted(embeddings_dict.keys())
    means = np.array([embeddings_dict[n].mean(axis=0) for n in names])
    sim = cosine_similarity(means)
    return pd.DataFrame(sim, index=names, columns=names)
