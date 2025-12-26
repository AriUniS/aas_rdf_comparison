import pandas as pd
import os
import tempfile
import re
import math
import hashlib
from datetime import datetime

import numpy as np
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import XSD

import base64
from rdflib.term import URIRef
from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from typing import Optional, Dict, List, Tuple



LIT = Namespace("urn:litfeat:")


def embed_ttl_graph(ttl_string: str):
    """
    Nimmt Turtle-Text und gibt ein embedding für jeden Knoten zurück.
    """
    # 1. Parse graph, to get entities / nodes
    g = rdflib.Graph()
    g.parse(data=ttl_string, format="turtle")

    entities = list({str(s) for s in g.subjects() if isinstance(s, URIRef)})

    ''' ausführlicher ist das:
    seen = set()
    entities = []
    for s in g.subjects():
        if isinstance(s, URIRef):
            uri = str(s)
            if uri not in seen:
                seen.add(uri)
                entities.append(uri)'''

    if not entities:
        return {}

    # 2. TTL in temporary file (pyRDF2Vec wants data path)
    with tempfile.NamedTemporaryFile(
        "w+", suffix=".ttl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(ttl_string)
        tmp_path = tmp.name

    try:
        # 3. Knowledge Graph from file
        kg = KG(location=tmp_path, fmt="turtle", is_remote=False)
        embeddings= [] #embeddings: List[np.ndarray] = []
        # 4. Transformer + Word2Vec (Workaround für 0.2.3-Bug)
        transformer = RDF2VecTransformer(
            Word2Vec(vector_size=100, sg=1, epochs=3),  # eig epochs 10
            #walkers=[],
            walkers=[RandomWalker(10, 10)]  # RandomWalker(2, 10) #RandomWalker(max_depth, max_walks)
        )
        embeddings, _ = transformer.fit_transform(kg, entities) # Einmal Walks extrahieren + Modell fitten + Embeddings für alle Entities
        # 6. Mapping URI -> Vektor (als Liste von floats)
        node_embeddings: Dict[str, List[float]] = {
            entity: embeddings[i].tolist()
            for i, entity in enumerate(entities)
        }

        return node_embeddings
        '''
        for e in entities:
            transformer = RDF2VecTransformer(
                Word2Vec(vector_size=100, sg=1, epochs=3),  # eig epochs 10
                walkers=[RandomWalker(2, 5)]               # RandomWalker(2, 10)
            )
            vecs, _ = transformer.fit_transform(kg, [e])
            embeddings.append(vecs[0])

        # 6. Mapping URI to Vektor (list)
        return {
            entity: embeddings[i].tolist()
            for i, entity in enumerate(entities)
        }'''
    finally:
        # delete temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

   # return node_embeddings
    # 6) Mapping URI to Vektor

def graph_embedding_from_nodes(node_vecs: dict[str, list[float]]) -> Optional[list[float]]: # ToDo hier weitermachen!!!
    """
        All Node-Embeddings to Graph-Embedding (Mean-Pooling).
        node_vecs: {URI: [emb_dim]}
        return: [emb_dim] oder None, falls kein Knoten. --> ToDo: PRÜFEN OB RICHTIG SO???!!!
        """
    if not node_vecs:
        return None

    # alle Vektoren in eine Matrix stapeln
    mat = np.vstack([np.array(v, dtype=float) for v in node_vecs.values()])
    graph_vec = mat.mean(axis=0)  # Mean-Pooling

    return graph_vec.tolist()


if __name__ == "__main__":  # Todo: Evtl raus, macht ja nichts!!!
    # Beispiel: ein Graph aus Datei
    with open("example_graph.ttl", "r", encoding="utf-8") as f:
        ttl_data = f.read()

    emb = embed_ttl_graph(ttl_data)

    print("Anzahl Knoten:", len(emb))
    first_key = next(iter(emb))
    print("Beispielvektor:", emb[first_key])