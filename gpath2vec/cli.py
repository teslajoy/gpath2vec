"""cli for gpath2vec: enrichment, network, embeddings."""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import click
import numpy as np
import pandas as pd
import scipy.sparse as sp

from gpath2vec.ea import enrich, ea_matrix, aggregate_min_fdr
from gpath2vec.net import Net
from gpath2vec.embedder import (
    PathwayMetapath2vec, SVDEmbedder, SpectralGraphEmbedder,
    LINEEmbedder, VAEEmbedder
)
from gpath2vec.aucell import compute_aucell, topk_per_niche

METHODS = ["metapath2vec", "svd", "spectral", "line", "vae"]


def _run_embedder(method, graph, ea_mat, study_id, dimensions, epochs, lr,
                  window, seed=1234):
    """run the selected embedding method, return embeddings dict."""
    if method == "metapath2vec":
        embedder = PathwayMetapath2vec(graph=graph, name=study_id, seed=seed)
        walks = embedder.model
        click.echo(f"{len(walks)} random walks")
        embedder.train_embeddings(walks=walks, dimensions=dimensions,
                                  window_size=window, epochs=epochs, lr=lr,
                                  seed=seed)
    elif method == "svd":
        if ea_mat is None:
            raise click.UsageError("svd requires an ea matrix (run enrichment first)")
        embedder = SVDEmbedder(ea_mat, dimensions=dimensions)
    elif method == "spectral":
        embedder = SpectralGraphEmbedder(graph, dimensions=dimensions)
    elif method == "line":
        embedder = LINEEmbedder(graph, dimensions=dimensions, epochs=epochs,
                                lr=lr, seed=seed)
    elif method == "vae":
        if ea_mat is None:
            raise click.UsageError("vae requires an ea matrix (run enrichment first)")
        embedder = VAEEmbedder(ea_mat, dimensions=dimensions, epochs=epochs,
                               lr=lr, seed=seed)
    else:
        raise click.UsageError(f"unknown method: {method}")

    return embedder


def _parse_genes(genes_str):
    """parse gene list from comma-separated string or file path."""
    if Path(genes_str).is_file():
        with open(genes_str) as f:
            return [line.strip() for line in f if line.strip()]
    return [g.strip() for g in genes_str.split(",") if g.strip()]


def _parse_gene_sets(gene_sets_str):
    """parse gene sets from json file or json string. returns {name: [genes]}."""
    p = Path(gene_sets_str)
    if p.is_file():
        with open(p) as f:
            return json.load(f)
    return json.loads(gene_sets_str)


def _make_id(study_id):
    return study_id or f"study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _outdir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


@click.group()
def cli():
    """convert biological pathways to embeddings with enrichment analysis"""
    pass


@cli.command("enrichment")
@click.option("--genes", required=True,
              help="comma-separated genes or path to file (one per line)")
@click.option("--gene-sets", required=False,
              help="json file or string with {name: [genes]} for multiple gene lists")
@click.option("--level", default="low", show_default=True,
              type=click.Choice(["high", "mid", "low", "all"]))
@click.option("--gene-filter", required=False,
              help="comma-separated genes to filter pathway universe (e.g. TF genes)")
@click.option("--weight", default="fdr", show_default=True,
              type=click.Choice(["fdr", "oddsratio"]),
              help="weight type for ea matrix")
@click.option("--min-genes", default=3, show_default=True)
@click.option("--out-path", required=True, help="output path (json)")
def perform_enrichment(genes, gene_sets, level, gene_filter, weight, min_genes, out_path):
    """run pathway enrichment analysis via fisher's exact test"""
    if gene_sets:
        gs = _parse_gene_sets(gene_sets)
    else:
        gene_list = _parse_genes(genes)
        gs = {"study": gene_list}

    gf = _parse_genes(gene_filter) if gene_filter else None

    click.echo(f"enrichment: {len(gs)} gene sets, level={level}")
    ea_df = enrich(gs, level=level, gene_filter=gf, min_genes=min_genes)

    if ea_df.empty:
        click.echo("no results")
        return

    sig = ea_df.sig_pathway.sum()
    click.echo(f"{len(ea_df)} tests, {sig} significant")

    # save ea dataframe as json
    _outdir(out_path)
    records = ea_df.to_dict(orient="records")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    # save ea matrix alongside
    matrix = ea_matrix(ea_df, weight=weight)
    matrix_path = out_path.replace(".json", "_matrix.csv")
    matrix.to_csv(matrix_path)
    click.echo(f"saved to {out_path}")
    click.echo(f"ea matrix ({matrix.shape[0]} x {matrix.shape[1]}) saved to {matrix_path}")


@cli.command("network")
@click.option("--enrichment-path", required=True, help="enrichment results json")
@click.option("--study-id", required=False)
@click.option("--level", default="all", show_default=True,
              type=click.Choice(["high", "mid", "low", "all"]))
@click.option("--gene-filter", required=False,
              help="comma-separated genes to filter pathway universe")
@click.option("--weight", default="fdr", show_default=True,
              type=click.Choice(["fdr", "oddsratio"]),
              help="weight type for cluster edges")
@click.option("--digraph", is_flag=True, default=True, show_default=True)
@click.option("--induce", is_flag=True, default=False, show_default=True)
@click.option("--out-path", required=True, help="output path (pickle)")
def create_network(enrichment_path, study_id, level, gene_filter, weight, digraph, induce, out_path):
    """create a pathway network from enrichment results"""
    assert Path(enrichment_path).is_file(), f"{enrichment_path} not found"
    study_id = _make_id(study_id)
    gf = _parse_genes(gene_filter) if gene_filter else None

    with open(enrichment_path) as f:
        ea_records = json.load(f)

    # min-fdr per pathway across all niches
    enrichment = aggregate_min_fdr(ea_records)

    # build cluster dict from ea records
    clusters = {}
    for r in ea_records:
        cname = r.get("cluster", "study")
        if not r.get("sig_pathway", False):
            continue
        if cname not in clusters:
            clusters[cname] = {}
        val = (1 - r["fdr_bh"]) if weight == "fdr" else r.get("oddsratio", 1.0)
        clusters[cname][r["stId"]] = val

    net = Net(enrichment=enrichment, id=study_id, digraph=digraph,
              induce=induce, level=level, gene_filter=gf,
              clusters=clusters if clusters else None)

    g = net.graph
    sig = sum(1 for _, a in g.nodes(data=True) if a.get("node_type") == "sig")
    n_clusters = sum(1 for _, a in g.nodes(data=True) if a.get("node_type") == "cluster")
    click.echo(f"network: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
               f"{sig} significant, {n_clusters} clusters")

    _outdir(out_path)
    net.save(out_path)
    click.echo(f"saved to {out_path}")


@cli.command("embeddings")
@click.option("--network-path", required=True, help="network pickle file")
@click.option("--ea-matrix-path", required=False, help="ea matrix csv (required for svd)")
@click.option("--method", default="metapath2vec", show_default=True,
              type=click.Choice(METHODS))
@click.option("--study-id", required=False)
@click.option("--dimensions", default=512, show_default=True)
@click.option("--window", default=5, show_default=True)
@click.option("--epochs", default=10, show_default=True)
@click.option("--lr", default=0.005, show_default=True)
@click.option("--seed", default=1234, show_default=True, type=int,
              help="rng seed for reproducible embeddings")
@click.option("--out-path", required=True, help="output path (pickle)")
@click.option("--save-model", required=False, help="optional model save path")
def generate_embeddings(network_path, ea_matrix_path, method, study_id,
                        dimensions, window, epochs, lr, seed, out_path,
                        save_model):
    """generate embeddings from network (metapath2vec, svd, spectral, line)"""
    import pandas as pd

    assert Path(network_path).is_file(), f"{network_path} not found"
    study_id = _make_id(study_id)

    net = Net(id=study_id)
    net.load(network_path)

    ea_mat = None
    if ea_matrix_path and Path(ea_matrix_path).is_file():
        ea_mat = pd.read_csv(ea_matrix_path, index_col=0)

    embedder = _run_embedder(method, net.graph, ea_mat, study_id,
                             dimensions, epochs, lr, window, seed)
    embeddings = embedder.get_embeddings()
    click.echo(f"embeddings for {len(embeddings)} nodes")

    _outdir(out_path)
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)
    click.echo(f"saved to {out_path}")

    if save_model:
        embedder.save_model(save_model)
        click.echo(f"model saved to {save_model}")


@cli.command("end2end")
@click.option("--genes", required=False,
              help="comma-separated genes or file path (single gene list)")
@click.option("--gene-sets", required=False,
              help="json file or string with {name: [genes]}")
@click.option("--output-dir", required=True)
@click.option("--study-id", required=False)
@click.option("--level", default="low", show_default=True,
              type=click.Choice(["high", "mid", "low", "all"]))
@click.option("--gene-filter", required=False,
              help="comma-separated genes to filter pathway universe")
@click.option("--weight", default="fdr", show_default=True,
              type=click.Choice(["fdr", "oddsratio"]))
@click.option("--method", default="metapath2vec", show_default=True,
              type=click.Choice(METHODS))
@click.option("--dimensions", default=512, show_default=True)
@click.option("--window", default=5, show_default=True)
@click.option("--epochs", default=10, show_default=True)
@click.option("--lr", default=0.005, show_default=True)
@click.option("--seed", default=1234, show_default=True, type=int,
              help="rng seed for reproducible embeddings")
@click.option("--n-jobs", default=1, show_default=True, type=int,
              help="parallel enrichment workers (order-preserving)")
@click.option("--reactome-dir", default=None,
              help="reactome cache dir (sets GPATH2VEC_REACTOME_DIR)")
@click.option("--meta", "meta_path", default=None,
              help="optional parquet to join into cluster embeddings; "
                   "all non-key columns are joined")
@click.option("--meta-key", default="cluster", show_default=True,
              help="key column in --meta matching cluster names")
def run_pipeline(genes, gene_sets, output_dir, study_id, level, gene_filter,
                 weight, method, dimensions, window, epochs, lr, seed,
                 n_jobs, reactome_dir, meta_path, meta_key):
    """run the full pipeline: enrichment -> network -> embeddings"""
    if reactome_dir:
        os.environ["GPATH2VEC_REACTOME_DIR"] = os.path.abspath(reactome_dir)
    study_id = _make_id(study_id)
    os.makedirs(output_dir, exist_ok=True)

    enrichment_path = os.path.join(output_dir, f"{study_id}_enrichment.parquet")
    matrix_path = os.path.join(output_dir, f"{study_id}_ea_matrix.csv")
    network_path = os.path.join(output_dir, f"{study_id}_network.pkl")
    embeddings_path = os.path.join(output_dir, f"{study_id}_embeddings.pkl")
    model_path = os.path.join(output_dir, f"{study_id}_model.pt")
    cluster_emb_path = os.path.join(
        output_dir, f"{study_id}_cluster_embeddings.parquet")
    provenance_path = os.path.join(
        output_dir, f"{study_id}_run_provenance.json")

    if gene_sets:
        gs = _parse_gene_sets(gene_sets)
    elif genes:
        gs = {"study": _parse_genes(genes)}
    else:
        raise click.UsageError("provide --genes or --gene-sets")

    gf = _parse_genes(gene_filter) if gene_filter else None

    # enrichment
    click.echo("step 1: enrichment")
    ea_df = enrich(gs, level=level, gene_filter=gf, n_jobs=n_jobs)
    ea_df.to_parquet(enrichment_path)
    records = ea_df.to_dict(orient="records")
    matrix = ea_matrix(ea_df, weight=weight)
    matrix.to_csv(matrix_path)

    # network
    click.echo("step 2: network")
    enrichment = aggregate_min_fdr(records)

    clusters = {}
    for r in records:
        cname = r.get("cluster", "study")
        if not r.get("sig_pathway", False):
            continue
        if cname not in clusters:
            clusters[cname] = {}
        val = (1 - r["fdr_bh"]) if weight == "fdr" else r.get("oddsratio", 1.0)
        clusters[cname][r["stId"]] = val

    net = Net(enrichment=enrichment, id=study_id, digraph=True,
              induce=False, level=level, gene_filter=gf,
              clusters=clusters if clusters else None)
    net.save(network_path)

    # embeddings
    click.echo(f"step 3: embeddings ({method})")
    embedder = _run_embedder(method, net.graph, matrix, study_id,
                             dimensions, epochs, lr, window, seed)
    embeddings = embedder.get_embeddings()
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    embedder.save_model(model_path)

    cluster_emb = {k: v for k, v in embeddings.items()
                   if isinstance(k, str) and k.startswith("cluster_")}
    if cluster_emb:
        emb_df = pd.DataFrame(cluster_emb).T
        emb_df.index = [i.replace("cluster_", "", 1) for i in emb_df.index]
        if meta_path:
            emb_df = _join_meta(emb_df, pd.read_parquet(meta_path), meta_key)
        emb_df.to_parquet(cluster_emb_path)

    prov = {
        "study_id": study_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": "end2end",
        "level": level, "weight": weight, "method": method,
        "dimensions": dimensions, "window": window, "epochs": epochs,
        "lr": lr, "seed": seed, "n_jobs": n_jobs,
        "gene_filter_applied": gf is not None,
        "meta_applied": meta_path is not None,
        "meta_key": meta_key if meta_path else None,
        "n_gene_sets": len(gs),
    }
    Path(provenance_path).write_text(json.dumps(prov, indent=2))

    click.echo(f"done. output in {output_dir}")
    click.echo(f"  enrichment: {enrichment_path}")
    click.echo(f"  ea matrix: {matrix_path}")
    click.echo(f"  network: {network_path}")
    click.echo(f"  embeddings: {embeddings_path}")
    if cluster_emb:
        click.echo(f"  cluster embeddings: {cluster_emb_path}")
    click.echo(f"  model: {model_path}")
    click.echo(f"  provenance: {provenance_path}")



def _load_gene_filter(path):
    """JSON list of gene symbols, or a dict with one list value."""
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, list):
        return set(map(str, obj))
    if isinstance(obj, dict):
        for k in ("genes", "tf_genes"):
            if isinstance(obj.get(k), list):
                return set(map(str, obj[k]))
        lists = [v for v in obj.values() if isinstance(v, list)]
        if len(lists) == 1:
            return set(map(str, lists[0]))
    raise click.ClickException(
        f"--gene-filter {path}: expected a JSON list of gene symbols "
        f"or a dict with one list value")


def _niche_row(X, i):
    """dense 1-D expression vector for niche row i (sparse or dense X)."""
    r = X[i]
    if sp.issparse(r):
        return np.asarray(r.todense()).ravel()
    return np.asarray(r).ravel()


def _join_meta(emb_df, meta, key):
    """join all non-key columns of `meta` into `emb_df` on its string index.
    no column names are hardcoded; whatever columns the user supplies in
    `meta` (besides `key`) end up as columns of `emb_df`. silent no-op if
    `meta` is None."""
    if meta is None:
        return emb_df
    if key not in meta.columns:
        raise click.ClickException(
            f"meta key {key!r} not in meta columns: {list(meta.columns)}")
    m = meta.copy()
    m[key] = m[key].astype(str)
    m = m.drop_duplicates(subset=[key]).set_index(key)
    for col in m.columns:
        emb_df[col] = emb_df.index.map(m[col].to_dict())
    return emb_df


@cli.command("niche-pipeline")
@click.option("--niche-matrix", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="niche x gene matrix: .npz (scipy sparse) or .npy (dense)")
@click.option("--genes", "genes_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help=".npy of gene symbols aligned to matrix columns")
@click.option("--niche-meta", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="parquet with a key column (default 'niche_id', set with "
                   "--niche-meta-key); any other columns are joined into the "
                   "cluster embeddings parquet")
@click.option("--out-dir", required=True, type=click.Path(file_okay=False))
@click.option("--reactome-dir", default=None,
              type=click.Path(exists=True, file_okay=False),
              help="reactome cache dir. REQUIRED on offline/HPC nodes; else "
                   "GPATH2VEC_REACTOME_DIR env or ~/.gpath2vec/cache")
@click.option("--enrichment", "enrichment_method", default="fisher",
              show_default=True, type=click.Choice(["fisher", "aucell"]))
@click.option("--reactome-level", default="low", show_default=True,
              type=click.Choice(["low", "mid", "high", "all"]))
@click.option("--gene-filter", "gene_filter_file", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="optional JSON gene-symbol list restricting the pathway "
                   "universe (off = full universe)")
@click.option("--min-genes", default=3, show_default=True, type=int)
@click.option("--max-genes", default=500, show_default=True, type=int,
              help="aucell: pathway gene-set size upper band")
@click.option("--top-genes", default=100, show_default=True, type=int,
              help="fisher: per-niche top-N expressed marker genes")
@click.option("--n-jobs", default=-1, show_default=True, type=int,
              help="fisher: parallel enrichment workers (-1 = all cores)")
@click.option("--topk", default=50, show_default=True, type=int,
              help="aucell: per-niche pathways kept (ablate 20/50/100)")
@click.option("--pre-normalized/--normalize", default=False,
              show_default=True,
              help="aucell: --normalize (default) applies "
                   "normalize_total(1e4)+log1p; --pre-normalized skips it")
@click.option("--dimensions", default=512, show_default=True, type=int)
@click.option("--epochs", default=5, show_default=True, type=int)
@click.option("--lr", default=0.005, show_default=True, type=float)
@click.option("--seed", default=1234, show_default=True, type=int,
              help="rng seed for reproducible walks + embeddings")
@click.option("--niche-meta-key", default="niche_id", show_default=True,
              help="key column in --niche-meta matching cluster names")
@click.option("--study-id", default=None)
def niche_pipeline(niche_matrix, genes_path, niche_meta, out_dir,
                   reactome_dir, enrichment_method, reactome_level,
                   gene_filter_file, min_genes, max_genes, top_genes,
                   n_jobs, topk, pre_normalized, dimensions, epochs, lr,
                   seed, niche_meta_key, study_id):
    """niche expression -> enrichment (fisher|aucell) -> graph -> embeddings.

    fisher: per-niche top-N expressed genes -> Fisher's exact vs Reactome,
    sig/notsig metapaths. aucell: full-ranking AUCell per niche -> per-niche
    top-k pathways as connectivity-typed edges, simplified metapaths. all
    paths are arguments; a provenance JSON is written for reproducibility.
    """
    if reactome_dir:
        os.environ["GPATH2VEC_REACTOME_DIR"] = os.path.abspath(reactome_dir)
    study_id = _make_id(study_id)
    os.makedirs(out_dir, exist_ok=True)

    mp = str(niche_matrix)
    if mp.endswith(".npz"):
        X = sp.load_npz(mp)
    elif mp.endswith(".npy"):
        X = np.load(mp, allow_pickle=False)
    else:
        raise click.ClickException(
            "--niche-matrix must be .npz (scipy sparse) or .npy (dense)")
    genes = np.load(genes_path, allow_pickle=True)
    meta = pd.read_parquet(niche_meta)
    if niche_meta_key not in meta.columns:
        raise click.ClickException(
            f"--niche-meta must have a {niche_meta_key!r} column "
            f"(set with --niche-meta-key)")
    niche_ids = meta[niche_meta_key].astype(str).tolist()
    if X.shape != (len(niche_ids), len(genes)):
        raise click.ClickException(
            f"shape mismatch: matrix {X.shape} vs "
            f"(n_niches={len(niche_ids)}, n_genes={len(genes)})")
    gene_filter = _load_gene_filter(gene_filter_file) if gene_filter_file else None
    click.echo(f"niche-pipeline: {len(niche_ids)} niches, {len(genes)} genes, "
               f"enrichment={enrichment_method}, level={reactome_level}")

    if enrichment_method == "fisher":
        gene_sets = {}
        for i, nid in enumerate(niche_ids):
            row = _niche_row(X, i)
            if (row > 0).sum() == 0:
                continue
            top = np.argsort(row)[-top_genes:]
            top = top[row[top] > 0]
            gene_sets[str(nid)] = list(map(str, genes[top]))
        click.echo(f"  {len(gene_sets)} niches with genes -> enrichment")
        ea_df = enrich(gene_sets, level=reactome_level,
                       gene_filter=gene_filter, min_genes=min_genes,
                       n_jobs=n_jobs)
        ea_df.to_parquet(os.path.join(out_dir, "enrichment.parquet"))
        clusters = {}
        for _, r in ea_df[ea_df.sig_pathway].iterrows():
            clusters.setdefault(str(r["cluster"]), {})[r["stId"]] = \
                1 - r["fdr_bh"]
        enrichment = aggregate_min_fdr(ea_df.to_dict("records"))
        net = Net(enrichment=enrichment, id=study_id, digraph=True,
                  level=reactome_level, gene_filter=gene_filter,
                  clusters=clusters if clusters else None)
        embedder = PathwayMetapath2vec(graph=net.graph, name=study_id,
                                       walks_per_node=10, walk_length=100,
                                       seed=seed)
    else:  # aucell
        scores = compute_aucell(
            X, genes, niche_ids, level=reactome_level,
            gene_filter=gene_filter, min_genes=min_genes,
            max_genes=max_genes, normalize=not pre_normalized,
            provenance_path=os.path.join(out_dir, "aucell_params.json"))
        scores.to_parquet(os.path.join(out_dir, "aucell_scores.parquet"))
        clusters = topk_per_niche(scores, topk)
        net = Net(enrichment=[], id=study_id, digraph=True,
                  level=reactome_level, gene_filter=gene_filter,
                  clusters=clusters if clusters else None,
                  node_typing="uniform")
        embedder = PathwayMetapath2vec(
            graph=net.graph, name=study_id, walks_per_node=10,
            walk_length=100, seed=seed,
            metapaths=[["cluster", "pathway", "pathway"],
                       ["pathway", "pathway", "pathway"]])

    g = net.graph
    n_clust = sum(1 for _, a in g.nodes(data=True)
                  if a.get("node_type") == "cluster")
    click.echo(f"  network: {g.number_of_nodes()} nodes, "
               f"{g.number_of_edges()} edges, {n_clust} niches")
    net.save(os.path.join(out_dir, "network.pkl"))

    embedder.train_embeddings(walks=embedder.model, dimensions=dimensions,
                              window_size=5, epochs=epochs, lr=lr, seed=seed)
    embeddings = embedder.get_embeddings()
    with open(os.path.join(out_dir, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings, f)
    embedder.save_model(os.path.join(out_dir, "model.pt"))

    cluster_emb = {k: v for k, v in embeddings.items()
                   if isinstance(k, str) and k.startswith("cluster_")}
    if cluster_emb:
        emb_df = pd.DataFrame(cluster_emb).T
        emb_df.index = [i.replace("cluster_", "", 1) for i in emb_df.index]
        emb_df = _join_meta(emb_df, meta, niche_meta_key)
        emb_df.to_parquet(
            os.path.join(out_dir, "cluster_embeddings.parquet"))

    prov = {
        "study_id": study_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "enrichment_method": enrichment_method,
        "reactome_level": reactome_level,
        "gene_filter_applied": gene_filter is not None,
        "min_genes": min_genes, "max_genes": max_genes,
        "top_genes_fisher": top_genes, "topk_aucell": topk,
        "pre_normalized": pre_normalized,
        "dimensions": dimensions, "epochs": epochs, "lr": lr,
        "seed": seed,
        "niche_meta_key": niche_meta_key,
        "n_niches": len(niche_ids), "n_genes": len(genes),
    }
    Path(os.path.join(out_dir, "run_provenance.json")).write_text(
        json.dumps(prov, indent=2))
    click.echo(f"done. output in {out_dir}")


def main():
    return cli()
