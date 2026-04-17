"""cli for gpath2vec: enrichment, network, embeddings."""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import click

from gpath2vec.ea import enrich, ea_matrix
from gpath2vec.net import Net
from gpath2vec.embedder import (
    PathwayMetapath2vec, SVDEmbedder, SpectralGraphEmbedder, LINEEmbedder
)

METHODS = ["metapath2vec", "svd", "spectral", "line"]


def _run_embedder(method, graph, ea_mat, study_id, dimensions, epochs, lr, window):
    """run the selected embedding method, return embeddings dict."""
    if method == "metapath2vec":
        embedder = PathwayMetapath2vec(graph=graph, name=study_id)
        walks = embedder.model
        click.echo(f"{len(walks)} random walks")
        embedder.train_embeddings(walks=walks, dimensions=dimensions,
                                  window_size=window, epochs=epochs, lr=lr)
    elif method == "svd":
        if ea_mat is None:
            raise click.UsageError("svd requires an ea matrix (run enrichment first)")
        embedder = SVDEmbedder(ea_mat, dimensions=dimensions)
    elif method == "spectral":
        embedder = SpectralGraphEmbedder(graph, dimensions=dimensions)
    elif method == "line":
        embedder = LINEEmbedder(graph, dimensions=dimensions, epochs=epochs, lr=lr)
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

    # build enrichment list for Net
    enrichment = []
    seen = set()
    for r in ea_records:
        stid = r["stId"]
        if stid not in seen:
            fdr = r.get("fdr_bh", r.get("entities", {}).get("fdr", 1.0))
            enrichment.append({"stId": stid, "entities": {"fdr": fdr}})
            seen.add(stid)

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
@click.option("--out-path", required=True, help="output path (pickle)")
@click.option("--save-model", required=False, help="optional model save path")
def generate_embeddings(network_path, ea_matrix_path, method, study_id,
                        dimensions, window, epochs, lr, out_path, save_model):
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
                             dimensions, epochs, lr, window)
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
def run_pipeline(genes, gene_sets, output_dir, study_id, level, gene_filter,
                 weight, method, dimensions, window, epochs, lr):
    """run the full pipeline: enrichment -> network -> embeddings"""
    study_id = _make_id(study_id)
    os.makedirs(output_dir, exist_ok=True)

    enrichment_path = os.path.join(output_dir, f"{study_id}_enrichment.json")
    matrix_path = os.path.join(output_dir, f"{study_id}_ea_matrix.csv")
    network_path = os.path.join(output_dir, f"{study_id}_network.pkl")
    embeddings_path = os.path.join(output_dir, f"{study_id}_embeddings.pkl")
    model_path = os.path.join(output_dir, f"{study_id}_model.pt")

    if gene_sets:
        gs = _parse_gene_sets(gene_sets)
    elif genes:
        gs = {"study": _parse_genes(genes)}
    else:
        raise click.UsageError("provide --genes or --gene-sets")

    gf = _parse_genes(gene_filter) if gene_filter else None

    # enrichment
    click.echo("step 1: enrichment")
    ea_df = enrich(gs, level=level, gene_filter=gf)
    records = ea_df.to_dict(orient="records")
    with open(enrichment_path, "w") as f:
        json.dump(records, f, indent=2)
    matrix = ea_matrix(ea_df, weight=weight)
    matrix.to_csv(matrix_path)

    # network
    click.echo("step 2: network")
    enrichment = []
    seen = set()
    for r in records:
        stid = r["stId"]
        if stid not in seen:
            enrichment.append({"stId": stid, "entities": {"fdr": r["fdr_bh"]}})
            seen.add(stid)

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
                             dimensions, epochs, lr, window)
    embeddings = embedder.get_embeddings()
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    embedder.save_model(model_path)

    click.echo(f"done. output in {output_dir}")
    click.echo(f"  enrichment: {enrichment_path}")
    click.echo(f"  ea matrix: {matrix_path}")
    click.echo(f"  network: {network_path}")
    click.echo(f"  embeddings: {embeddings_path}")
    click.echo(f"  model: {model_path}")



def main():
    return cli()
