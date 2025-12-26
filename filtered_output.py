import argparse
import requests
import pandas as pd

from backend_paths import FUSEKI_ENDPOINT

from urllib.parse import quote
import matplotlib.pyplot as plt

from rdf2vec_application import embed_ttl_graph, graph_embedding_from_nodes
from typing import List, Dict, Tuple
from Visualization.visualize_embeddings import visualize_all_embeddings
from scipy.spatial import distance

'''
Dilemma hier: SPARQL-Eintragung als 2-stufeigen Prozess oder direkt --> ToDo!!!!!!
WICHTIG!!!
'''


def run_select(endpoint: str, query: str, auth=None):
    """schickt eine SPARQL-SELECT an Fuseki und gibt JSON zurück"""
    url = endpoint.rstrip("/") + "/query"
    resp = requests.post(
        url,
        data={"query": query},
        headers={"Accept": "application/sparql-results+json"},
        timeout=30,
        auth=auth,
    )
    resp.raise_for_status()
    return resp.json()

def _try_get(endpoint: str, graph_uri: str, path: str, fmt: str, auth=None):
    base = endpoint.rstrip("/")
    url = f"{base}/{path}?graph={quote(graph_uri, safe='')}"
    resp = requests.get(url, headers={"Accept": fmt}, timeout=30, auth=auth)
    return resp

def download_graph_2(endpoint: str, graph_uri: str, fmt="text/turtle", auth=None):
    """
    holt einen benannten Graphen vom Fuseki-Dataset.
    1. Versuch: /data?graph=...
    2. Fallback: /get?graph=...
    """
    # 1. Versuch
    resp = _try_get(endpoint, graph_uri, "data", fmt, auth=auth)
    if resp.status_code == 200:
        return resp.text

    # 2. Versuch
    resp2 = _try_get(endpoint, graph_uri, "get", fmt, auth=auth)
    if resp2.status_code == 200:
        return resp2.text

    # Wenn beides nicht klappt: saubere Fehlermeldung
    raise requests.HTTPError(
        f"Konnte Graph {graph_uri} nicht laden. "
        f"/data lieferte {resp.status_code}, /get lieferte {resp2.status_code}."
    )

def download_graph(endpoint: str, graph_uri: str, fmt="text/turtle", auth=None):
    """
    holt einen benannten Graphen vom Fuseki-Dataset.
    1. Versuch: /data?graph=...
    2. Fallback: /get?graph=...
    """
    # 1. Versuch
    resp = _try_get(endpoint, graph_uri, "data", fmt, auth=auth)
    if resp.status_code == 200:
        return resp.text

    # 2. Versuch
    resp2 = _try_get(endpoint, graph_uri, "get", fmt, auth=auth)
    if resp2.status_code == 200:
        return resp2.text

    # Wenn beides nicht klappt: saubere Fehlermeldung
    raise requests.HTTPError(
        f"Konnte Graph {graph_uri} nicht laden. "
        f"/data lieferte {resp.status_code}, /get lieferte {resp2.status_code}."
    )

    '''
    base = endpoint.rstrip("/")
    graph_url = f"{base}/data?graph={quote(graph_uri, safe='')}"
    resp = requests.get(graph_url, headers={"Accept": fmt}, timeout=30)
    resp.raise_for_status()
    return resp.text
    '''


'''
Todo: Prüfen: Ist das Argument hier richtig???!!!
'''
def enable_visualization(all_embeddings: List[Dict[str, object]]) -> None:
    visualize_all_embeddings(all_embeddings)

def select_graphs(
    endpoint: str,
    sparql: str,
    var_name: str = "g",
    auth=None,
) -> List[str]:
    """
    Führt eine SPARQL-SELECT-Query aus und gibt eine Liste
    der gefundenen Graph-URIs (ohne Duplikate) in stabiler Reihenfolge zurück.
    """
    data = run_select(endpoint, sparql, auth=auth)
    graphs: List[str] = []
    seen = set()

    for binding in data["results"]["bindings"]:
        if var_name not in binding:     #ToDo: Das muss ich UNBEDINGT abändern!!!
            continue  # andere Bindings interessieren uns nicht
        g = binding[var_name]["value"]
        if g not in seen:
            seen.add(g)
            graphs.append(g)

    return graphs


def embed_graphs(
    endpoint: str,
    graph_uris: List[str],
    auth=None,
) -> Tuple[List[Dict[str, object]], Dict[str, List[float]]]:
    """
    Lädt die angegebenen Graph-URIs vom Fuseki-Endpoint, berechnet RDF2Vec-
    Embeddings und gibt:
      - eine Liste ALLER Embeddings (Nodes + Graph) und
      - ein Dict {graph_uri -> graph_vector}
    zurück.
    """
    all_embeddings: List[Dict[str, object]] = []
    graph_vectors: Dict[str, List[float]] = {}

    for g in graph_uris:
        try:
            ttl = download_graph_2(endpoint, g, fmt="text/turtle", auth=auth)
        except requests.HTTPError as e:
            print(f"lala Fehler beim Laden von {g}: {e}")
            continue

        node_vecs = embed_ttl_graph(ttl)
        if not node_vecs:
            print(f"Keine URI-Subjekte in {g} gefunden – keine Embeddings erzeugt.")
            continue

        # Node-Embeddings
        for node_uri, vec in node_vecs.items():
            all_embeddings.append({
                "kind": "node",
                "graph_uri": g,
                "id": node_uri,
                "vec": vec,
            })

        # Graph-Embedding (Mittelwert)
        graph_vec = graph_embedding_from_nodes(node_vecs)
        if graph_vec is not None:
            all_embeddings.append({
                "kind": "graph",
                "graph_uri": g,
                "id": g,
                "vec": graph_vec,
            })
            graph_vectors[g] = graph_vec

    return all_embeddings, graph_vectors



def main():     # Wichtig: Das wird nicht in der UI aufgerufen, sondern ist nur für Konsolennutzung interessant
    # ToDo: Spaeter RAUSMACHEN: Für Codecleaning
    parser = argparse.ArgumentParser(
        description="SPARQL gegen Fuseki ausführen, Graph-URIs aus dem Resultat holen und die Graphen abrufen."
    )
    parser.add_argument(
        "--endpoint",
        default=FUSEKI_ENDPOINT,
        help="Fuseki-Dataset-URL, z.B. http://localhost:3030/aas"
        f"(Default: {FUSEKI_ENDPOINT})",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query-file", help="Datei mit SPARQL-Query")
    group.add_argument("--query", help="SPARQL-Query als String")
    parser.add_argument(
        "--var-name",
        default="g",
        help="Name der SELECT-Variable, die die Graph-URI enthält (Default: g)",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="geladene Graphen auf stdout ausgeben",
    )
    parser.add_argument(
        "--user",
        help="Username für Basic Auth (falls Fuseki geschützt ist)",
    )
    parser.add_argument(
        "--password",
        help="Passwort für Basic Auth (falls Fuseki geschützt ist)",
    )
    args = parser.parse_args()
    auth = None
    if args.user and args.password:
        auth = (args.user, args.password)

    # SPARQL laden
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            sparql = f.read()
    else:
        sparql = args.query

    all_graph_embeddings = []   #Liste

    # 1. Query ausführen
    data = run_select(args.endpoint, sparql)

    # 2. Graph-URIs aus dem Resultat holen
    graphs = set()
    for binding in data["results"]["bindings"]:
        if args.var_name not in binding:
            continue  # Query hat andere Bindings, die uns nicht interessieren
        g = binding[args.var_name]["value"]
        graphs.add(g)

    if not graphs:
        print("Keine Graphen gefunden.")
        return

    print("Gefundene Graphen:")
    for g in graphs:
        print(" -", g)

    # COLLECTOR FOR ALL EMBEDDINGS (Nodes + Graphs)
    all_embeddings: List[Dict[str, object]] = []

    # 3. Jeden gefundenen Graphen abrufen
    for g in graphs:
        try:
            ttl = download_graph(args.endpoint, g, fmt="text/turtle", auth=auth)

        except requests.HTTPError as e:
            print(f"Fehler beim Laden von {g}: {e}")
            continue

        print(f"\n=== Graph {g} ===")
        print(f"- TTL-Länge: {len(ttl)} Zeichen")


        # RDF2Vec-NODE-Embeddings
        node_vecs = embed_ttl_graph(ttl)
        if not node_vecs:
            print("Keine URI-Subjekte gefunden – keine Embeddings erzeugt.")
            continue

        print(f"- Eingebettete Knoten: {len(node_vecs)}")
        print(f"Node Vectors:{node_vecs}")
        # ALLE NODE-EMBEDDINGS SAMMELN
        for node_uri, vec in node_vecs.items():
            all_embeddings.append({
                "kind": "node",
                "graph_uri": g,
                "id": node_uri,
                "vec": vec,
            })

        # 2) Graph-Embedding (Mittelwert)
        graph_vec = graph_embedding_from_nodes(node_vecs)
        print(f"Graph Vector: {graph_vec}")

        if graph_vec is None:
            print("Kein Graph-Embedding erzeugt.")
            continue

        # GRAPH-EMBEDDING
        all_embeddings.append({
            "kind": "graph",
            "graph_uri": g,
            "id": g,
            "vec": graph_vec,
        })


    graph_vectors: list = []

    for i in all_embeddings:
        if i["kind"]=="graph":
            graph_vectors.append(i["vec"])
    dist1 = distance.cosine(graph_vectors[0], graph_vectors[1])
    dist2 = distance.cosine(graph_vectors[0], graph_vectors[2])
    dist3 = distance.cosine(graph_vectors[1], graph_vectors[2])

    print(dist1)
    print(dist2)
    print(dist3)


if __name__ == "__main__":
    main()





'''
        # 4. Alles gemeinsam visualisieren
        if all_embeddings:
            visualize_all_embeddings(all_embeddings)
        else:
            print("\nKeine Embeddings erzeugt – nichts zu visualisieren.")
'''
        # Wenn du den Vektor sehen willst:
        # print(node_vecs[first_uri])
       # else:
            # hier kann man statt print z.B. speichern oder mit rdflib weiterarbeiten # ToDo: Hier weitermachen
         #   print(f"Graph {g} erfolgreich geladen ({len(ttl)} Zeichen).")
            #print(ttl)  # Darin ist die ttl gespeichert --> damit weiterarbeiten