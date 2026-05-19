"""AUCell per-niche pathway scoring, as an alternative enrichment source.

WHAT THIS COMPUTES (read this before using it)
----------------------------------------------
The unit of analysis is the *niche*: an upstream-defined spatial group of cells
(e.g. a center spot plus its spatial neighbors), summarized as one aggregated
"pseudobulk" expression vector per niche. The input here is therefore an
(n_niches x n_genes) matrix where each row is that niche's aggregated counts.

For each niche, AUCell:
  1. ranks all genes by the niche's aggregated expression (descending),
  2. for each Reactome pathway gene set, walks down that ranking and integrates
     the recovery curve (how early the pathway's genes appear) over the top
     `n_up` features. With decoupler's default (n_up=None) the recovery window
     is the top 5% of ranked genes by magnitude.
  3. returns one continuous score in [0, ~1] per (niche, pathway): high = the
     pathway's genes are concentrated among the niche's most-expressed genes.

There is no per-niche gene-set selection step (no top-100, no MAD/mean): AUCell
uses the *full* gene ranking, so the "which genes count per niche" choice that
Fisher's exact requires does not exist here. The only parameters are the
gene-set size band and the AUC recovery window (n_up); both are logged.

This is a faithful generalization of the validated niche_aucell.py pipeline:
same normalization (normalize_total to 1e4 then log1p), same decoupler call
(`dc.mt.aucell`, tmin = min gene-set size), same gene-set-size band. It is
generalized by (a) taking all paths/params as arguments, (b) drawing the
pathway universe from `ea.filter_pathways` so the AUCell and Fisher enrichment
paths use the IDENTICAL level-filtered Reactome universe (comparable by
construction), and (c) writing a provenance JSON for the methods section.

decoupler + anndata are imported lazily; install via the `aucell` extra.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import ea


def _sha256_of_array(X) -> str:
    """stable content hash of the niche x gene matrix, for provenance."""
    h = hashlib.sha256()
    if sp.issparse(X):
        Xc = X.tocsr()
        for a in (Xc.data, Xc.indices, Xc.indptr,
                  np.asarray(Xc.shape, dtype=np.int64)):
            h.update(np.ascontiguousarray(a).tobytes())
    else:
        h.update(np.ascontiguousarray(np.asarray(X)).tobytes())
    return h.hexdigest()


def _build_net(level, gene_filter, min_genes, max_genes):
    """long-form pathway->gene table for decoupler, drawn from the SAME
    `ea.filter_pathways` universe the Fisher path uses. `source` is the bare
    Reactome stId so the resulting score columns are drop-in compatible with
    `Net` (which keys pathway nodes by stId)."""
    gm = ea.filter_pathways(level=level, gene_filter=gene_filter,
                            min_genes=min_genes)
    if gm.empty:
        raise ValueError(f"no pathways for level={level!r} "
                         f"(gene_filter set: {gene_filter is not None})")
    rows = []
    kept = 0
    for r in gm.itertuples(index=False):
        g = list(r.genes)
        # gene-set-size band: min_genes (also passed to decoupler as tmin)
        # and an upper cap to drop very large, non-specific sets.
        if min_genes <= len(g) <= max_genes:
            kept += 1
            for sym in g:
                rows.append((r.stId, sym))
    if not rows:
        raise ValueError(
            f"no pathways within size band [{min_genes}, {max_genes}] "
            f"at level={level!r}")
    net = pd.DataFrame(rows, columns=["source", "target"])
    return net, kept


def compute_aucell(X, genes, niche_ids, *, level="all", gene_filter=None,
                   min_genes=3, max_genes=500, normalize=True,
                   provenance_path=None, verbose=True):
    """per-niche AUCell scores against the level-filtered Reactome universe.

    Parameters
    ----------
    X : (n_niches, n_genes) array or scipy.sparse
        per-niche aggregated (pseudobulk) expression. Raw counts by default;
        pass `normalize=False` if already library-normalized + log1p'd.
    genes : sequence of str, length n_genes
        gene symbols aligned to the columns of X.
    niche_ids : sequence, length n_niches
        identifier per row of X (used as the output index).
    level : {"low","mid","high","all"}
        Reactome pathway level (same semantics as ea.filter_pathways).
    gene_filter : optional set/list of gene symbols
        restrict the pathway universe (e.g. TF genes). None = no restriction.
    min_genes, max_genes : int
        pathway gene-set size band. min_genes is also passed to decoupler as
        `tmin`. Defaults [3, 500] match the validated pipeline.
    normalize : bool
        if True (default) apply normalize_total(target_sum=1e4) then log1p
        in-place, exactly as the validated niche_aucell pipeline. The score
        ranking is invariant to a per-niche scalar, but this is kept for
        parity with the reference and is logged.
    provenance_path : optional path
        if given, write a JSON of every parameter (decoupler version, n_up
        policy, tmin/tmax, normalization, input sha256, shapes) for the
        methods section / reproducibility.

    Returns
    -------
    aucell_df : DataFrame (n_niches x n_pathways), index = niche_ids,
        columns = Reactome stIds, values = continuous AUCell scores.
    """
    try:
        import anndata as ad
        import decoupler as dc
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "AUCell requires 'decoupler' and 'anndata'. Install the extra: "
            "pip install 'gpath2vec[aucell]'") from e

    genes = list(map(str, genes))
    niche_ids = list(niche_ids)
    if sp.issparse(X):
        X = X.tocsr()
    if X.shape != (len(niche_ids), len(genes)):
        raise ValueError(
            f"X shape {X.shape} does not match "
            f"(n_niches={len(niche_ids)}, n_genes={len(genes)})")

    input_sha = _sha256_of_array(X)

    adata = ad.AnnData(
        X=X.tocsr() if sp.issparse(X) else np.asarray(X),
        obs=pd.DataFrame({"niche_id": niche_ids}),
        var=pd.DataFrame(index=genes),
    )
    adata.obs_names = [str(n) for n in niche_ids]
    adata.obs_names_make_unique()

    if normalize:
        # normalize_total(target_sum=1e4) then log1p, done explicitly so the
        # exact transform is auditable and matches the reference pipeline.
        if verbose:
            print("  normalizing: target_sum=1e4 -> log1p")
        Xc = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
        lib = np.asarray(Xc.sum(axis=1)).ravel()
        scaling = 1e4 / np.maximum(lib, 1.0)
        Xc = Xc.multiply(scaling[:, None]).tocsr()
        Xc.data = np.log1p(Xc.data)
        adata.X = Xc

    net, n_path = _build_net(level, gene_filter, min_genes, max_genes)
    if verbose:
        print(f"  pathway net: {net['source'].nunique()} pathways, "
              f"{len(net)} pathway-gene edges (level={level}, "
              f"size band [{min_genes}, {max_genes}])")
        print("  running AUCell (decoupler)...")

    # tmin = minimum gene-set size; n_up left at decoupler default (None ->
    # top 5% of ranked features). Both recorded in provenance below.
    dc.mt.aucell(adata, net=net, tmin=min_genes, verbose=verbose)
    scores = adata.obsm["score_aucell"].copy()
    scores.index = niche_ids[: len(scores)] if len(scores) == len(niche_ids) \
        else scores.index

    if provenance_path is not None:
        prov = {
            "method": "AUCell",
            "decoupler_version": dc.__version__,
            "anndata_version": ad.__version__,
            "n_up": None,
            "n_up_policy": ("decoupler default: top 5% of ranked features "
                            "by magnitude"),
            "tmin": int(min_genes),
            "tmax_genesize_band": int(max_genes),
            "reactome_level": level,
            "gene_filter_applied": gene_filter is not None,
            "n_gene_filter": (len(set(gene_filter))
                              if gene_filter is not None else 0),
            "normalize_total_target_sum": 1e4 if normalize else None,
            "log1p": bool(normalize),
            "pre_normalized_input": (not normalize),
            "n_niches": int(len(niche_ids)),
            "n_genes": int(len(genes)),
            "n_pathways_scored": int(scores.shape[1]),
            "input_matrix_sha256": input_sha,
        }
        Path(provenance_path).write_text(json.dumps(prov, indent=2))
        if verbose:
            print(f"  wrote AUCell provenance -> {provenance_path}")

    return scores


def topk_per_niche(aucell_df, k):
    """top-k AUCell pathways per niche -> clusters dict for `Net`.

    aucell_df : DataFrame (n_niches x n_pathways), continuous AUCell scores.
    k : pathways to keep per niche (the primary sparsity parameter; ablate
        over e.g. {20, 50, 100}). Low k = sparse, sharp per-niche identity but
        risk of under-connected niches; high k = denser, smoother, re-approaches
        the everything-connects-to-everything dilution. Always ablate + log k.

    Mechanics: each niche row is the aggregated pseudobulk of its constituent
    cells; its AUCell vector is that niche's pathway-activity profile. Keeping
    the k highest-scoring pathways (dropping zeros) makes the niche->pathway
    graph edges, with the RAW AUCell score as edge weight (non-negative, in
    [0, ~1], directly usable by the weighted random walker).

    returns: {niche_id: {pathway_stId: aucell_score, ...}} -- identical shape
    to the Fisher path's clusters dict, drop-in for the Net builder.
    """
    clusters = {}
    for niche_id in aucell_df.index:
        scores = aucell_df.loc[niche_id]
        top = scores.nlargest(k)
        top = top[top > 0]  # drop zero-AUCell pathways even if within top-k
        clusters[str(niche_id)] = top.to_dict()
    return clusters