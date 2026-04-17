"""pathway enrichment via fisher's exact test with fdr correction."""

import io
import tarfile
import zipfile

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.multitest as multi

from . import utils

np.random.seed(1234)


def gene_mappings():
    """stId -> name + gene list from ReactomePathways.gmt.zip"""
    raw = utils.fetch("ReactomePathways.gmt.zip",
                      "https://reactome.org/download/current/ReactomePathways.gmt.zip",
                      binary=True)
    if raw is None:
        return []
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        lines = zf.open(zf.infolist()[0]).readlines()
    return [
        dict(name=(p := line.decode("utf8").strip().split("\t"))[0],
             stId=p[1], genes=p[2:])
        for line in lines
    ]


def ehld_stids():
    """high-level pathway stIds (ehld)."""
    text = utils.fetch("svgsummary.txt",
                       "https://reactome.org/download/current/ehld/svgsummary.txt")
    return [s for s in (text or "").splitlines() if "R-" in s]


def sbgn_stids():
    """low-level pathway stIds (sbgn minus ehld)."""
    raw = utils.fetch("homo_sapiens.sbgn.tar.gz",
                      "https://reactome.org/download/current/homo_sapiens.sbgn.tar.gz",
                      binary=True)
    if raw is None:
        return []
    names = tarfile.open(fileobj=io.BytesIO(raw)).getnames()
    ehlds = set(ehld_stids())
    return [n.replace(".sbgn", "") for n in names if n.replace(".sbgn", "") not in ehlds]


def level_stids(level):
    """return set of stIds for a given level: high, mid, low, or all."""
    if level == "all":
        return None
    if level == "high":
        return set(ehld_stids())
    if level == "low":
        return set(sbgn_stids())
    if level == "mid":
        all_stids = set(r["stId"] for r in gene_mappings())
        return all_stids - set(ehld_stids()) - set(sbgn_stids())
    raise ValueError(f"unknown level: {level}")


def filter_pathways(level="all", gene_filter=None, min_genes=3):
    """
    filter pathway universe before EA.

    level: high/mid/low/all
    gene_filter: optional set of genes to restrict which pathways are included
    min_genes: minimum pathway size
    returns: filtered gene_mappings dataframe
    """
    mappings = gene_mappings()
    if not mappings:
        return pd.DataFrame(columns=["name", "stId", "genes"])
    gm = pd.json_normalize(mappings)

    stids = level_stids(level)
    if stids is not None:
        gm = gm[gm.stId.isin(stids)]

    if gene_filter is not None:
        gf = set(gene_filter)
        gm = gm[gm.genes.apply(lambda g: bool(set(g) & gf))]

    gm = gm[gm.genes.apply(len) > min_genes].copy()
    return gm


def enrich(gene_sets, level="low", gene_filter=None, min_genes=3):
    """
    fisher's exact test per gene-set/pathway pair with fdr correction.

    gene_sets: {name: [genes]}
    level: high/mid/low/all
    gene_filter: optional set of genes to restrict pathway universe
    min_genes: minimum pathway size to include
    returns: dataframe with cluster, stId, name, oddsratio, pvalue, fdr_bh
    """
    gm = filter_pathways(level=level, gene_filter=gene_filter, min_genes=min_genes)
    if gm.empty:
        return pd.DataFrame()

    universe = sorted(set(g for genes in gm.genes for g in genes))
    u_set = set(universe)
    pw_genes = {row.stId: set(row.genes) & u_set for _, row in gm.iterrows()}
    n_pw = len(pw_genes)
    n_universe = len(universe)

    results = []
    for cname, cgenes in gene_sets.items():
        cset = set(cgenes) & u_set
        c_in = len(cset)
        c_out = n_universe - c_in

        rows = []
        for stid, pset in pw_genes.items():
            a = len(cset & pset)
            b = len(pset) - a
            c = c_in - a
            d = c_out - b
            oddsratio, pvalue = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(dict(cluster=str(cname), stId=stid,
                             oddsratio=oddsratio, pvalue=pvalue))

        edf = pd.DataFrame(rows)
        corrected = multi.multipletests(edf.pvalue, alpha=0.05, method="fdr_bh",
                                        is_sorted=False, returnsorted=False)
        edf["fdr_bh"] = corrected[1]
        edf["sig_pathway"] = corrected[0]
        edf["pathway_adjpvalue"] = -np.log10(edf.fdr_bh) / n_pw
        edf.fillna(value={"pathway_adjpvalue": 1}, inplace=True)
        results.append(edf)

    ea_df = pd.concat(results, ignore_index=True)
    ea_df["name"] = ea_df.stId.map(gm.set_index("stId")["name"])
    return ea_df


def ea_matrix(ea_df, weight="fdr", sig_only=True):
    """
    pivot enrichment results into cluster x pathway matrix.

    weight: "fdr" (uses 1 - fdr_bh) or "oddsratio"
    sig_only: if true, zero out non-significant entries
    returns: dataframe with clusters as rows, pathway stIds as columns
    """
    df = ea_df.copy()
    if weight == "fdr":
        df["value"] = 1 - df.fdr_bh
    elif weight == "oddsratio":
        df["value"] = df.oddsratio
    else:
        raise ValueError(f"unknown weight: {weight}, expected fdr/oddsratio")

    if sig_only:
        df.loc[~df.sig_pathway, "value"] = 0.0

    matrix = df.pivot_table(index="cluster", columns="stId", values="value", fill_value=0.0)
    return matrix
