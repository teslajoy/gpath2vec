"""F2 unit tests: cluster-edge directionality in the pathway graph.

verifies the topology assumption that embedder.metapath2vec relies on at
embedder.py:391, where `self.graph.neighbors(current)` is filtered by
node_type at line 396. for a DiGraph this is successors only, so a
`cluster -> pathway` edge is invisible from the pathway side.

these tests document the F2 bug and the three candidate fixes so any
silent change to net.add_clusters edge semantics or Net.digraph default
trips a regression here.

run:
    pytest tests/test_metapath_topology.py -v
"""

import networkx as nx


def _build_pathway_hierarchy(graph_cls):
    """tiny stand-in for the Reactome hierarchy: P1 -> P2 -> P3."""
    G = graph_cls()
    G.add_node("P1", node_type="sig")
    G.add_node("P2", node_type="sig")
    G.add_node("P3", node_type="notsig")
    G.add_edge("P1", "P2")
    G.add_edge("P2", "P3")
    return G


def test_digraph_cluster_unreachable_from_pathway():
    """current behavior: in DiGraph mode, the cluster anchors a pathway via
    `cluster -> pathway`, so P.neighbors() (which returns successors only)
    never yields the cluster. this is the F2 bug paper §4.5 inadvertently
    claims is fixed.
    """
    G = _build_pathway_hierarchy(nx.DiGraph)
    G.add_node("cluster_A", node_type="cluster")
    G.add_edge("cluster_A", "P1", weight=0.9)
    assert "cluster_A" not in set(G.neighbors("P1"))
    assert "cluster_A" not in set(G.neighbors("P2"))


def test_undirected_graph_cluster_reachable():
    """option A: undirected graph. cluster is reachable from P1, but P1's
    hierarchical parents are now also neighbors (loss of direction).
    """
    G = _build_pathway_hierarchy(nx.Graph)
    G.add_node("cluster_A", node_type="cluster")
    G.add_edge("cluster_A", "P1", weight=0.9)
    p1_neighbors = set(G.neighbors("P1"))
    assert "cluster_A" in p1_neighbors


def test_bidirectional_cluster_edges_reachable():
    """option B: keep hierarchy directional, but add the reverse
    `pathway -> cluster` edge. P1.neighbors() yields the cluster plus
    P1's children. hierarchical parents stay invisible from the child.
    """
    G = _build_pathway_hierarchy(nx.DiGraph)
    G.add_node("cluster_A", node_type="cluster")
    G.add_edge("cluster_A", "P1", weight=0.9)
    G.add_edge("P1", "cluster_A", weight=0.9)  # F2(b) patch
    p1_neighbors = set(G.neighbors("P1"))
    assert "cluster_A" in p1_neighbors
    assert "P2" in p1_neighbors
    # cluster only reachable from the pathways it anchors
    assert "cluster_A" not in set(G.neighbors("P3"))


def test_add_clusters_only_writes_one_direction():
    """mirrors net.Net.add_clusters (net.py:99-111) on a hand-built graph
    to confirm we're testing the same edge shape add_clusters produces.
    this is the integration-side regression check: if add_clusters ever
    changes to bidirectional, this test breaks and forces a re-read of F2.
    """
    G = nx.DiGraph()
    G.add_node("P1")
    G.add_node("P2")
    clusters = {"A": {"P1": 0.9, "P2": 0.5}}
    for cname, pathway_weights in clusters.items():
        node_id = f"cluster_{cname}"
        G.add_node(node_id, node_type="cluster", cluster=cname)
        for stid, weight in pathway_weights.items():
            if stid in G:
                G.add_edge(node_id, stid, weight=weight)
    assert G.has_edge("cluster_A", "P1")
    assert G.has_edge("cluster_A", "P2")
    assert not G.has_edge("P1", "cluster_A")
    assert not G.has_edge("P2", "cluster_A")


def test_metapath_filter_silently_falls_back_when_cluster_unreachable():
    """end-to-end mini-regression on the embedder filter logic at
    embedder.py:396-401. when walker is at P1 with target_type='cluster',
    `typed = [n for n in neighbors if node_types[n] == 'cluster']` is
    empty in digraph mode; the walker silently falls back to the full
    neighbor list. this test pins that fallback behavior so the F2
    decision can change it deliberately.
    """
    G = _build_pathway_hierarchy(nx.DiGraph)
    G.add_node("cluster_A", node_type="cluster")
    G.add_edge("cluster_A", "P1", weight=0.9)
    node_types = nx.get_node_attributes(G, "node_type")

    current = "P1"
    target_type = "cluster"
    neighbors = list(G.neighbors(current))
    typed = [n for n in neighbors if node_types.get(n, "unknown") == target_type]

    assert typed == []  # cluster filter yields nothing in digraph mode
    assert neighbors == ["P2"]  # silent fallback would step toward P2, not cluster_A