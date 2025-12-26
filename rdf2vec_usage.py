# rdf2vec_application.py
import os
import re
import math
import hashlib
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import XSD

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker


LIT = Namespace("urn:litfeat:")


# ----------------------------
# Literal-AUGMENTATION (ohne Ersetzen)
# ----------------------------
def _hash_id(*parts: str, n: int = 16) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def _num_bin(x: float) -> str:
    ax = abs(x)
    if ax == 0:
        return "NUM_0"
    exp = int(math.floor(math.log10(ax)))
    exp = max(-6, min(6, exp))  # clamp
    return f"NUM_1e{exp}"


def _len_bin(n: int) -> str:
    if n <= 5:
        return "LEN_1_5"
    if n <= 20:
        return "LEN_6_20"
    if n <= 100:
        return "LEN_21_100"
    return "LEN_101_PLUS"


def _text_shape(s: str) -> str:
    s = s.strip()
    if re.match(r"^https?://", s):
        return "TEXT_URL"
    if "@" in s and "." in s:
        return "TEXT_EMAIL"
    if re.match(r"^\+?\d[\d\s\-]{5,}$", s):
        return "TEXT_PHONE"
    if re.match(r"^\d+(\.\d+)?$", s):
        return "TEXT_NUMERIC_STRING"
    if re.match(r"^[A-Za-z0-9_\-]+$", s):
        return "TEXT_ALPHANUM"
    return "TEXT_FREE"


def _try_parse_date(lex: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(lex.replace("Z", "+00:00"))
    except Exception:
        return None


def augment_literals(g: Graph) -> Graph:
    """
    Gibt einen neuen Graphen zurück:
      - enthält alle Originaltriples
      - fügt zusätzlich (s,p,litNode) + litNode-Features hinzu
    => Literale beeinflussen Walks/Embeddings, ohne dass du Literalwerte aufzählen musst.
    """
    g2 = Graph()
    for t in g:
        g2.add(t)

    for s, p, o in list(g.triples((None, None, None))):
        if not isinstance(o, Literal):
            continue

        lex = str(o)
        dt_uri = str(o.datatype) if o.datatype else ""
        lang = o.language or ""

        lit_node = URIRef(f"urn:lit:{_hash_id(str(s), str(p), lex, dt_uri, lang)}")

        # zusätzliche Kante (Original bleibt erhalten!)
        g2.add((s, p, lit_node))

        # Kind + Feature-Tokens (endlich!)
        kind = "OTHER"

        # Numerics
        if o.datatype in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                          XSD.decimal, XSD.double, XSD.float):
            kind = "NUM"
            try:
                x = float(lex)
                g2.add((lit_node, LIT.numBin, URIRef(LIT[_num_bin(x)])))
            except Exception:
                pass

        # Date / DateTime
        elif o.datatype in (XSD.date, XSD.dateTime):
            kind = "DATE"
            dt = _try_parse_date(lex)
            if dt:
                g2.add((lit_node, LIT.year, URIRef(LIT[f"Y_{dt.year}"])))
                g2.add((lit_node, LIT.month, URIRef(LIT[f"YM_{dt.year}_{dt.month:02d}"])))

        # Duration
        elif o.datatype == XSD.duration:
            kind = "DURATION"
            # robust, ohne vollständiges ISO-8601 Parsing:
            g2.add((lit_node, LIT.durShape, URIRef(LIT["DUR_ISO8601"])))

        # Strings / langStrings
        else:
            # language-tag oder xsd:string/None -> TEXT
            if lang or (o.datatype in (None, XSD.string)):
                kind = "TEXT"
                g2.add((lit_node, LIT.textShape, URIRef(LIT[_text_shape(lex)])))
                g2.add((lit_node, LIT.lenBin, URIRef(LIT[_len_bin(len(lex))])))
                if lang:
                    g2.add((lit_node, LIT.lang, URIRef(LIT[f"LANG_{lang}"])))

        # Grundfeatures
        g2.add((lit_node, LIT.kind, URIRef(LIT[kind])))
        if dt_uri:
            # datatype als kompaktes Token
            g2.add((lit_node, LIT.datatype, URIRef(LIT[_hash_id(dt_uri, n=10)])))

    return g2


# ----------------------------
# Joint RDF2Vec Training (ein Vektorraum)
# ----------------------------
def _entities_for_graph(g: Graph) -> List[str]:
    """
    Entities, die wir später für den Graph-Vektor poolen:
    - alle URI subjects + URI objects (inkl. litNodes, weil wir sie als URIs erzeugen)
    """
    ents = set()
    for s in g.subjects():
        if isinstance(s, URIRef):
            ents.add(str(s))
    for o in g.objects():
        if isinstance(o, URIRef):
            ents.add(str(o))
    return sorted(ents)


def graph_embedding_from_nodes(node_vecs: Dict[str, List[float]]) -> Optional[List[float]]:
    """
    Mean pooling über Knotenvektoren (stabiler Graph-Vektor).
    """
    if not node_vecs:
        return None
    mat = np.vstack([np.array(v, dtype=float) for v in node_vecs.values()])
    return mat.mean(axis=0).tolist()


def embed_graphs_from_ttls(
    ttl_by_uri: Dict[str, str],
    vector_size: int = 100,
    max_depth: int = 10,
    max_walks: int = 10,
    epochs: int = 5,
    seed: int = 42,
) -> Tuple[Dict[str, Dict[str, List[float]]], Dict[str, List[float]]]:
    """
    WICHTIG:
    - trainiert EIN gemeinsames RDF2Vec/Word2Vec Modell über ALLE Graphen
    - Literale werden via augment_literals() einbezogen (ohne Auflistung)
    Rückgabe:
      all_node_embeddings_by_graph: {graph_uri: {entity_uri: vec}}
      graph_vectors: {graph_uri: graph_vec}
    """

    # Reproducibility (so gut es geht)
    np.random.seed(seed)

    parsed_aug_graphs: Dict[str, Graph] = {}
    entities_by_graph: Dict[str, List[str]] = {}

    # 1) Parse + literal augmentation
    for graph_uri, ttl in ttl_by_uri.items():
        g = Graph()
        g.parse(data=ttl, format="turtle")
        g_aug = augment_literals(g)
        parsed_aug_graphs[graph_uri] = g_aug
        entities_by_graph[graph_uri] = _entities_for_graph(g_aug)

    # 2) Merge all graphs into one training KG
    g_all = Graph()
    for g in parsed_aug_graphs.values():
        for t in g:
            g_all.add(t)

    entities_all = sorted(set(e for ents in entities_by_graph.values() for e in ents))
    if not entities_all:
        return {}, {}

    # 3) Write merged graph to temp file (pyrdf2vec KG expects a file path)
    with tempfile.NamedTemporaryFile("w+", suffix=".ttl", delete=False, encoding="utf-8") as tmp:
        tmp.write(g_all.serialize(format="turtle"))
        tmp_path = tmp.name

    try:
        kg = KG(location=tmp_path, fmt="turtle", is_remote=False)

        transformer = RDF2VecTransformer(
            Word2Vec(vector_size=vector_size, sg=1, epochs=epochs),
            walkers=[RandomWalker(max_depth, max_walks, with_reverse=True)],
        )

        emb_matrix, _ = transformer.fit_transform(kg, entities_all)
        emb_global: Dict[str, List[float]] = {
            entities_all[i]: emb_matrix[i].tolist()
            for i in range(len(entities_all))
        }

        # 4) Per-graph node embeddings + pooled graph vectors
        all_node_embeddings_by_graph: Dict[str, Dict[str, List[float]]] = {}
        graph_vectors: Dict[str, List[float]] = {}

        for graph_uri, ents in entities_by_graph.items():
            node_vecs = {e: emb_global[e] for e in ents if e in emb_global}
            all_node_embeddings_by_graph[graph_uri] = node_vecs

            gv = graph_embedding_from_nodes(node_vecs)
            if gv is not None:
                graph_vectors[graph_uri] = gv

        return all_node_embeddings_by_graph, graph_vectors

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
