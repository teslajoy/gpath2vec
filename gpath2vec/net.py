"""reactome pathway network construction."""

import pickle
from itertools import chain

import networkx as nx

from . import utils
from . import ea


def fetch_human_hierarchy():
    """parent-child pathway relations for homo sapiens, local cache or http."""
    text = utils.fetch("ReactomePathwaysRelation.txt",
                       "https://reactome.org/download/current/ReactomePathwaysRelation.txt")
    if text is None:
        return []
    return [tuple(line.split("\t")) for line in text.splitlines() if "-HSA" in line]


class Net:
    """
    pathway network over the reactome homo sapiens hierarchy.

    enrichment: list of dicts [{"stId": ..., "entities": {"fdr": ...}}]
    level: high/mid/low/all (filters pathways by ehld/sbgn before building)
    gene_filter: optional set of genes to restrict pathway universe
    clusters: optional dict {cluster_name: {stId: weight}} to add
              gene lists of interest (e.g. clusters) as nodes connected
              to their significant pathways with ea weights
    """

    def __init__(self, enrichment=None, id="study", digraph=False,
                 induce=False, level="all", gene_filter=None,
                 clusters=None):
        self.enrichment = enrichment or []
        self.digraph = digraph
        self.induce = induce
        self.id = id
        self.level = level
        self.gene_filter = gene_filter
        self.pathway_relations = fetch_human_hierarchy()
        self.pathway_stids = set(chain(*self.pathway_relations))
        self.graph = self._build_graph()
        self.fdr_values = [p.get("entities", {}).get("fdr", 1.0) for p in self.enrichment]
        self.ea_stids = [p["stId"] for p in self.enrichment]
        self._set_node_attr()
        if clusters is not None:
            self.add_clusters(clusters)

    def _build_graph(self):
        G = nx.DiGraph(study=self.id) if self.digraph else nx.Graph(study=self.id)
        G.add_edges_from(self.pathway_relations)

        if self.level != "all":
            keep = ea.level_stids(self.level)
            if keep is not None:
                G = G.subgraph(n for n in G.nodes() if n in keep).copy()

        if self.gene_filter is not None:
            gm = ea.filter_pathways(level=self.level, gene_filter=self.gene_filter)
            keep = set(gm.stId)
            G = G.subgraph(n for n in G.nodes() if n in keep).copy()

        if self.induce and self.enrichment:
            sig = [p["stId"] for p in self.enrichment
                   if p.get("entities", {}).get("fdr", 1) < 0.05]
            G = G.subgraph(sig).copy()

        return G

    def _set_node_attr(self):
        fdr_vals = self.fdr_values
        id_keys = self.ea_stids

        attr = {
            id_keys[i]: {
                "het": 1 if fdr_vals[i] < 0.05 else 0,
                "features": [1 - round(fdr_vals[i], 4)],
                "feature": round(fdr_vals[i], 4),
                "node_type": "sig" if fdr_vals[i] < 0.05 else "notsig",
                "stId": id_keys[i],
            }
            for i in range(len(id_keys))
        }

        not_ea = self.pathway_stids - set(id_keys)
        attr_not_ea = {
            stid: {"het": -1, "features": [-1], "feature": -1,
                   "node_type": "notsig", "stId": stid}
            for stid in not_ea
        }

        nx.set_node_attributes(self.graph, utils.pathway_parent_mappings(), "parent_pathway")
        nx.set_node_attributes(self.graph, utils.pathway_name_mappings(), "pathway_name")
        nx.set_node_attributes(self.graph, attr)
        nx.set_node_attributes(self.graph, attr_not_ea)

    def add_clusters(self, clusters):
        """
        add gene lists of interest (clusters) as nodes in the graph,
        connected to their significant pathways with ea weights.

        clusters: {cluster_name: {stId: weight}}
        """
        for cname, pathway_weights in clusters.items():
            node_id = f"cluster_{cname}"
            self.graph.add_node(node_id, node_type="cluster", cluster=cname)
            for stid, weight in pathway_weights.items():
                if stid in self.graph:
                    self.graph.add_edge(node_id, stid, weight=weight)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self, path):
        # only load graphs you trust, pickle can execute arbitrary code
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
