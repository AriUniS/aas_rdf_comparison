from rdflib.plugins.sparql.parser import parseQuery
from filtered_output import select_graphs
from typing import Optional


def filter_graphs(fuseki_endpoint: str, sparql_query: Optional[str]= None) -> list[str]:
    """
    Liefert eine Liste von Graph-URIs aus Fuseki.

    - Wenn sparql_query leer ist -> default: alle Graphen.
    - Wenn sparql_query gesetzt ist -> Syntaxcheck + Ausführung.
    - UI-frei: wirft Exceptions, statt ui.notify.
    """
    raw = (sparql_query or "").strip()
    DEFAULT_LIST_GRAPHS_QUERY = """
    SELECT DISTINCT ?g WHERE {
      GRAPH ?g { }
    }
    """

    # Fall 1: leer -> alle Graphen
    if not raw:
        return select_graphs(fuseki_endpoint, DEFAULT_LIST_GRAPHS_QUERY)

    # Fall 2: Query vorhanden -> Syntax prüfen + ausführen
    parseQuery(raw)  # wirft Exception bei Syntaxfehler
    return select_graphs(fuseki_endpoint, raw)


