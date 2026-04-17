"""shared utilities for fetching and caching reactome data."""

import os
import json
from pathlib import Path

import requests
from requests.exceptions import ConnectionError, RequestException


def _cache_dir():
    """
    return the cache directory for reactome data.
    priority: GPATH2VEC_REACTOME_DIR env var > ~/.gpath2vec/cache
    """
    d = os.environ.get("GPATH2VEC_REACTOME_DIR")
    if d:
        return Path(d)
    default = Path.home() / ".gpath2vec" / "cache"
    default.mkdir(parents=True, exist_ok=True)
    return default


# keep this for backwards compat with ea.py / net.py
_local_dir = _cache_dir


def fetch(filename, url, binary=False):
    """
    fetch a file from url, caching locally on first download.
    returns text or bytes depending on binary flag. returns None on failure.
    """
    cache = _cache_dir()
    cached = cache / filename

    if cached.exists():
        return cached.read_bytes() if binary else cached.read_text()

    try:
        r = requests.get(url=url)
        r.raise_for_status()
    except (ConnectionError, RequestException) as e:
        print(f"[gpath2vec] {e}")
        return None

    # save to cache
    if binary:
        cached.write_bytes(r.content)
    else:
        cached.write_text(r.text)

    return r.content if binary else r.text


def get_event_hierarchy(species="9606"):
    """fetch the full event hierarchy for a species from reactome."""
    cached = _cache_dir() / f"events_hierarchy_{species}.json"
    if cached.exists():
        with open(cached) as f:
            return json.load(f)

    url = f"https://reactome.org/ContentService/data/eventsHierarchy/{species}"
    try:
        r = requests.get(url=url, headers={"accept": "application/json"})
        r.raise_for_status()
    except (ConnectionError, RequestException) as e:
        print(f"[gpath2vec] could not fetch event hierarchy: {e}")
        return None

    data = r.json()
    with open(cached, "w") as f:
        json.dump(data, f)
    return data


_hierarchy_cache = None


def _hierarchy():
    global _hierarchy_cache
    if _hierarchy_cache is None:
        _hierarchy_cache = get_event_hierarchy(species="9606")
    return _hierarchy_cache


def get_json_items(json_obj, key):
    if isinstance(json_obj, dict):
        for k, v in json_obj.items():
            if k == key:
                yield v
            elif isinstance(v, (dict, list)):
                yield from get_json_items(v, key)
    elif isinstance(json_obj, list):
        for item in json_obj:
            yield from get_json_items(item, key)


def pathway_parent_mappings():
    hierarchy = _hierarchy()
    if hierarchy is None:
        return {}
    parent = [p["name"] for p in hierarchy]
    pathways = [list(set(get_json_items(p, "stId"))) for p in hierarchy]
    for i in range(len(pathways)):
        pathways[i].append(hierarchy[i]["stId"])
    pathway_mappings = {parent[i]: pathways[i] for i in range(len(parent))}
    return {v: k for k, values in pathway_mappings.items() for v in values}


def pathway_name_mappings():
    """stId -> pathway name mapping."""
    text = fetch("ReactomePathways.txt",
                 "https://reactome.org/download/current/ReactomePathways.txt")
    if text is None:
        return {}

    entities = {}
    for line in text.splitlines():
        if "-HSA" not in line:
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            entities[parts[0]] = parts[1]
    return entities
