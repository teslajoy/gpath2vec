"""
Command-line interface for Pathway2Vec.

1. Performing pathway enrichment analysis
2. Creating pathway networks
3. Generating pathway embeddings
"""

import sys
import os
import json
import pickle
import requests
import click
from pathlib import Path
from datetime import datetime

from pathway2vec.embedder import PathwayMetapath2vec
from pathway2vec.net import Net


@click.group()
def cli():
    """Convert biological pathways to embeddings with enrichment analysis"""
    pass


@cli.command('enrichment')
@click.option('--genes', required=True,
              help='Comma-separated list of genes or path to file with one gene per line')
@click.option('--organism', required=False,
              default='human',
              show_default=True,
              help='Organism for enrichment analysis')
@click.option('--out_path', required=True,
              help='Path to save enrichment results - JSON format')
@click.option('--threshold', required=False,
              default=0.05,
              help='Enrichment analysis FDR p-value threshold')
@click.option('--verbose', is_flag=True, required=False,
              default=False,
              show_default=True)
def perform_enrichment(genes, organism, threshold, out_path, verbose):
    """perform pathway enrichment analysis using g:Profiler"""
    gene_list = []
    if Path(genes).is_file():
        with open(genes, 'r') as f:
            gene_list = [line.strip() for line in f if line.strip()]
    else:
        gene_list = [g.strip() for g in genes.split(',') if g.strip()]

    if verbose:
        click.echo(f"Performing enrichment analysis for {len(gene_list)} genes...")

    api_url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
    payload = {
        "organism": "hsapiens" if organism == "human" else organism,
        "query": gene_list,
        "sources": ["REAC"],
        "user_threshold": threshold,
        "all_results": True,
        "ordered": False,
        "no_iea": False,
        "measure_underrepresentation": False,
        "domain_scope": "annotated",
        "significance_threshold_method": "g_SCS"
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        click.echo(f"Error during enrichment analysis: {e}", err=True)
        sys.exit(1)

    pathway_results = []
    if "result" in result and result["result"]:
        for item in result["result"]:
            if item["source"] == "REAC":
                reactome_id = item["native"].replace("REAC:", "")
                pathway_result = {
                    "stId": reactome_id,
                    "entities": {
                        "fdr": item["p_value"],
                        "genes": item["intersections"]
                    },
                    "name": item["name"],
                    "pValue": item["p_value"],
                    "significant": item["p_value"] < threshold
                }
                pathway_results.append(pathway_result)

    pathway_results.sort(key=lambda x: x["entities"]["fdr"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(pathway_results, f, indent=2)

    significant_count = sum(1 for p in pathway_results if p["significant"])
    click.echo(f"Found {len(pathway_results)} Reactome pathways, {significant_count} significant (FDR < 0.05)")
    click.echo(f"Enrichment results saved to {out_path}")


@cli.command('network')
@click.option('--enrichment_path', required=True,
              help='Path to enrichment results JSON file')
@click.option('--study_id', required=False,
              help='Study identifier (defaults to timestamp-based ID)')
@click.option('--digraph', is_flag=True, required=False,
              default=True,
              show_default=True,
              help='Create directed graph')
@click.option('--induce', is_flag=True, required=False,
              default=False,
              show_default=True,
              help='Induce subgraph of significant pathways')
@click.option('--study', is_flag=True, required=False,
              default=True,
              show_default=True,
              help='Include study nodes and edge weights')
@click.option('--out_path', required=True,
              help='Path to save network (pickle format)')
@click.option('--verbose', is_flag=True, required=False,
              default=False,
              show_default=True)
def create_network(enrichment_path, study_id, digraph, induce, study, out_path, verbose):
    """create a pathway network from enrichment results"""
    assert Path(enrichment_path).is_file(), f"Path {enrichment_path} is not a valid file path."

    if not study_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_id = f"study_{timestamp}"

    with open(enrichment_path, 'r') as f:
        enrichment_results = json.load(f)

    if verbose:
        click.echo(f"Creating pathway network for study {study_id}...")

    network = Net(
        enrichment=enrichment_results,
        id=study_id,
        digraph=digraph,
        induce=induce,
        study=study
    )

    graph = network.graph
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    sig_count = sum(1 for _, attr in graph.nodes(data=True)
                    if "node_type" in attr and attr["node_type"] == "sig")

    if verbose:
        click.echo(f"Network created with {node_count} nodes and {edge_count} edges")
        click.echo(f"Significant pathways: {sig_count}")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    network.save_graph(out_path)
    click.echo(f"Network saved to {out_path}")


@cli.command('embeddings')
@click.option('--network_path', required=True,
              help='Path to network pickle file')
@click.option('--study_id', required=False,
              help='Study identifier (defaults to timestamp-based ID)')
@click.option('--dimensions', required=False,
              default=128,
              show_default=True,
              help='Embedding dimensions')
@click.option('--window', required=False,
              default=5,
              show_default=True,
              help='Context window size')
@click.option('--epochs', required=False,
              default=10,
              show_default=True,
              help='Training epochs')
@click.option('--out_path', required=True,
              help='Path to save embeddings (pickle format)')
@click.option('--save_model', required=False,
              help='Path to save full Word2Vec model (optional)')
@click.option('--verbose', is_flag=True, required=False,
              default=False,
              show_default=True)
def generate_embeddings(network_path, study_id, dimensions, window, epochs, out_path, save_model, verbose):
    """generate pathway embeddings from network"""
    assert Path(network_path).is_file(), f"Path {network_path} is not a valid file path."

    if not study_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_id = f"study_{timestamp}"

    network = Net(
        enrichment=[],
        id=study_id,
        digraph=True,
        induce=False,
        study=True
    )
    network.load_graph(network_path)

    if verbose:
        click.echo(f"Generating pathway embeddings for study {study_id}...")

    embedder = PathwayMetapath2vec(
        graph=network.graph,
        name=study_id
    )

    walks = embedder.model
    if verbose:
        click.echo(f"Generated {len(walks)} random walks")

    if verbose:
        click.echo(f"Training Word2Vec model (dimensions={dimensions}, window={window}, epochs={epochs})...")

    model = embedder.train_embeddings(
        walks=walks,
        dimensions=dimensions,
        window_size=window,
        epochs=epochs
    )

    embeddings = embedder.get_embeddings()
    embedding_count = len(embeddings)
    if verbose:
        click.echo(f"Generated embeddings for {embedding_count} pathways")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, 'wb') as f:
        pickle.dump(embeddings, f)
    click.echo(f"Embeddings saved to {out_path}")

    if save_model:
        embedder.save_model(save_model)
        click.echo(f"Word2Vec model saved to {save_model}")


@cli.command('pipeline')
@click.option('--genes', required=True,
              help='Comma-separated list of genes or path to file with one gene per line')
@click.option('--output_dir', required=True,
              help='Directory to save all output files')
@click.option('--study_id', required=False,
              help='Study identifier (defaults to timestamp-based ID)')
@click.option('--dimensions', required=False,
              default=128,
              show_default=True,
              help='Embedding dimensions')
@click.option('--window', required=False,
              default=5,
              show_default=True,
              help='Context window size')
@click.option('--epochs', required=False,
              default=10,
              show_default=True,
              help='Training epochs')
@click.option('--organism', required=False,
              default='human',
              show_default=True,
              help='Organism for enrichment analysis')
@click.option('--verbose', is_flag=True, required=False,
              default=False,
              show_default=True)
def run_pipeline(genes, output_dir, study_id, dimensions, window, epochs, organism, verbose):
    """run the complete Pathway2Vec pipeline (enrichment → network → embeddings)"""
    if not study_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_id = f"study_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    enrichment_path = os.path.join(output_dir, f"{study_id}_enrichment.json")
    network_path = os.path.join(output_dir, f"{study_id}_network.pkl")
    embeddings_path = os.path.join(output_dir, f"{study_id}_embeddings.pkl")
    model_path = os.path.join(output_dir, f"{study_id}_model.model")

    click.echo("Step 1: performing pathway enrichment analysis...")
    ctx = click.Context(perform_enrichment)
    ctx.invoke(
        perform_enrichment,
        genes=genes,
        organism=organism,
        out_path=enrichment_path,
        verbose=verbose
    )

    click.echo("\nStep 2: creating pathway network...")
    ctx = click.Context(create_network)
    ctx.invoke(
        create_network,
        enrichment_path=enrichment_path,
        study_id=study_id,
        digraph=True,
        induce=False,
        study=True,
        out_path=network_path,
        verbose=verbose
    )

    ctx = click.Context(generate_embeddings)
    ctx.invoke(
        generate_embeddings,
        network_path=network_path,
        study_id=study_id,
        dimensions=dimensions,
        window=window,
        epochs=epochs,
        out_path=embeddings_path,
        save_model=model_path,
        verbose=verbose
    )

    click.echo(f"\nPipeline completed successfully! All results saved to {output_dir}")
    click.echo(f"  • Enrichment results: {enrichment_path}")
    click.echo(f"  • Network: {network_path}")
    click.echo(f"  • Embeddings: {embeddings_path}")
    click.echo(f"  • model: {model_path}")


if __name__ == "__main__":
    cli()