from pathlib import Path

import os

FUSEKI_BASE = os.environ.get("FUSEKI_BASE", "http://localhost:3030")
FUSEKI_DATASET = os.environ.get("FUSEKI_DATASET", "aas")

# vollständiger Endpoint für SELECT/DATA/GET
FUSEKI_ENDPOINT = f"{FUSEKI_BASE.rstrip('/')}/{FUSEKI_DATASET}"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'

RAW_JSON_INIT_DIR = DATA_DIR / 'raw' / 'init'
RAW_JSON_LIB_DIR = DATA_DIR / 'raw' / 'lib'
TTL_INIT_DIR = DATA_DIR / 'ttl' / 'init'
TTL_LIB_DIR = DATA_DIR / 'ttl' / 'lib'
EMB_GRAPH_DIR = DATA_DIR / 'embeddings' / 'graphs'

for d in [RAW_JSON_INIT_DIR, RAW_JSON_LIB_DIR, TTL_INIT_DIR, TTL_LIB_DIR, EMB_GRAPH_DIR]:
    d.mkdir(parents=True, exist_ok=True)