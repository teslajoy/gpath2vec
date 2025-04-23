import requests
from requests.exceptions import ConnectionError
import networkx as nx
from itertools import chain
from . import utils

def fetch_human_hierarchy() -> list:
    """
    Fetches Reactome pathways hierarchy for Homo sapiens.

    Returns:
        List of pathway relations.
    """
    url = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"

    try:
        response = requests.get(url=url)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except ConnectionError as e:
        print(e)
        return []

    if response.status_code == 200:
        content_list = response.text.splitlines()
        relations = [tuple(d.split("\t")) for d in content_list if '-HSA' in d]
        return relations
    else:
        print('Status code returned a value of %s' % response.status_code)
        return []

class Net:
    def __init__(self, enrichment: list, id: str, digraph: bool = False, induce: bool = False, study: bool = False):
        self.enrichment = enrichment
        self.digraph = digraph
        self.induce = induce
        self.id = id
        self.study = study
        self.pathway_relations = fetch_human_hierarchy()
        self.pathway_stids = self.human_pathway_stids()
        self.graph = self.reactome_net()
        self.fdr_values = self.get_fdrs()
        self.ea_stids = self.get_ea_stids()
        self.node_attr = self.node_attr()
        self.study_node_and_edge_weight()

    def human_pathway_stids(self):
        pathway_stids = [stid for stid in chain(*self.pathway_relations)]
        return set(pathway_stids)

    def reactome_net(self):
        reactome_G = nx.DiGraph(study=self.id) if self.digraph else nx.Graph(study=self.id)

        if self.induce and self.enrichment:
            reactome_G = reactome_G.subgraph([p['stId'] for p in self.enrichment if p['fdr'] < 0.05])

        reactome_G.add_edges_from(self.pathway_relations)
        return reactome_G

    def get_ea_stids(self):
        return [p['stId'] for p in self.enrichment]

    def get_fdrs(self):
        return [p['entities']['fdr'] for p in self.enrichment]

    def node_attr(self):
        id_keys = self.ea_stids
        fdr_vals = self.fdr_values

        # Calculate 'het' and 'node_type' lists
        het = [1 if f < 0.05 else 0 for f in fdr_vals]
        node_type = ["sig" if f < 0.05 else "notsig" for f in fdr_vals]

        attr = {
            id_keys[i]: {
                "het": het[i],
                "features": [1 - round(fdr_vals[i], 4)],
                "feature": round(fdr_vals[i], 4),
                "node_type": node_type[i],
                "stId": id_keys[i]
            }
            for i in range(len(id_keys))
        }

        id_keys_not_ea = list(self.pathway_stids - set(id_keys))
        attr_not_ea = {
            id_keys_not_ea[i]: {
                "het": -1,
                "features": [-1],
                "feature": -1,
                "node_type": "notsig",
                "stId": id_keys_not_ea[i]
            }
            for i in range(len(id_keys_not_ea))
        }

        pathway_mappings_nodes = utils.pathway_parent_mappings()
        pathway_names = utils.pathway_name_mappings()

        nx.set_node_attributes(self.graph, pathway_mappings_nodes, "parent_pathway")
        nx.set_node_attributes(self.graph, pathway_names, "pathway_name")
        nx.set_node_attributes(self.graph, attr)
        nx.set_node_attributes(self.graph, attr_not_ea)

    def study_node_and_edge_weight(self):
        id_keys = self.ea_stids
        fdr_vals = [1 - round(f, 4) for f in self.fdr_values]

        if self.study:
            relations = [
                ("_".join([id_keys[i], self.id]), id_keys[i], {'weight': fdr_vals[i]})
                for i in range(len(id_keys))
            ]
            self.graph.add_edges_from(relations)
        else:
            pass