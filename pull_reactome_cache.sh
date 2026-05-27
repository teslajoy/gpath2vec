#!/usr/bin/env bash
# pre-stage the Reactome cache gpath2vec needs.

set -euo pipefail

CACHE_DIR="${1:-$HOME/.gpath2vec/cache}"
PYTHON="${PYTHON:-python}"

mkdir -p "$CACHE_DIR"
# resolve to an absolute path so the env var is unambiguous on other nodes
CACHE_DIR="$(cd "$CACHE_DIR" && pwd)"
export GPATH2VEC_REACTOME_DIR="$CACHE_DIR"

echo "interpreter : $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "cache dir   : $CACHE_DIR"
echo "warming Reactome cache (requires internet)..."

"$PYTHON" - <<'PY'
# call the package's real fetchers; each populates one cached file.
from gpath2vec import ea, net, utils

steps = [
    ("ReactomePathways.gmt.zip",      ea.gene_mappings),
    ("ehld/svgsummary.txt",           ea.ehld_stids),
    ("homo_sapiens.sbgn.tar.gz",      ea.sbgn_stids),
    ("ReactomePathwaysRelation.txt",  net.fetch_human_hierarchy),
    ("ReactomePathways.txt",          utils.pathway_name_mappings),
    ("events_hierarchy_9606.json",    lambda: utils.get_event_hierarchy("9606")),
    ("pathway parent mappings",       utils.pathway_parent_mappings),
]
for label, fn in steps:
    print(f"  fetching: {label} ...", flush=True)
    fn()
print("all package fetchers completed.")
PY

echo
echo "cache contents:"
ls -la "$CACHE_DIR"
if [ -z "$(ls -A "$CACHE_DIR" 2>/dev/null)" ]; then
    echo "ERROR: cache dir is empty after warm-up" >&2
    exit 1
fi
echo
echo "done. on the compute node either:"
echo "  export GPATH2VEC_REACTOME_DIR='$CACHE_DIR'"
echo "  # or pass --reactome-dir '$CACHE_DIR' to gpath2vec niche-pipeline"