#!/usr/bin/env python3
"""
Lädt alle .ttl-Dateien aus einem Verzeichnis in Apache Jena Fuseki.

Konfiguration:
- per CLI:
    python upload_aas_graphs.py --fuseki http://localhost:3030 --dataset aas --dir AAS_Upload
- oder per ENV:
    FUSEKI_BASE, FUSEKI_DATASET, AAS_DIR

Falls nichts angegeben wird:
- Fuseki: http://localhost:3030
- Dataset: aas
- Verzeichnis: ./AAS_Upload
"""

# ToDo: if idshort vergeben: Nummerierte ab _1 --> Damit man nicht doppelte Graphen quasi in einen Graphen bekommt!!!

import os
import sys
import argparse
import urllib.parse
import requests
from pathlib import Path

from typing import Dict, Union, Iterable, Optional

try:
    import rdflib
except ImportError:
    print("[FEHLER] rdflib nicht installiert. Bitte 'pip install rdflib requests' ausführen.")
    sys.exit(1)

from backend_paths import FUSEKI_BASE, FUSEKI_DATASET, FUSEKI_ENDPOINT

AAS_NS = "https://admin-shell.io/aas/3/0/"
QUERY_IDSHORT = f"""
PREFIX aas: <{AAS_NS}>
SELECT ?idShort WHERE {{
  ?aas a aas:AssetAdministrationShell ;
       <{AAS_NS}Referable/idShort> ?idShort .
}}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="TTL-Dateien nach Fuseki hochladen.")
    parser.add_argument(
        "--fuseki",
        default=FUSEKI_BASE,
        help="Basis-URL von Fuseki, z.B. http://localhost:3030",
    )
    parser.add_argument(
        "--dataset",
        default=FUSEKI_DATASET,
        help="Name of Fuseki Dataset",
    )
    parser.add_argument(
        "--dir",
        default=os.environ.get("AAS_DIR", "AAS_Upload"),
        #default=os.environ.get("AAS_DIR", "AAS_Upload", "data/ttl/init", "data/ttl/lib"),
        help="Verzeichnis mit .ttl-Dateien",
    )
    parser.add_argument(
        "--user",
        default="admin", #os.environ.get("ADMIN_USER"), --> Todo: Das automatisieren, ist gepfuscht
        help="Fuseki-Username (falls Basic Auth aktiv ist)",
    )
    parser.add_argument(
        "--password",
        default="admin",#os.environ.get("ADMIN_PASSWORD"),--> Todo: Das automatisieren, ist gepfuscht
        help="Fuseki-Passwort (falls Basic Auth aktiv ist)",
    )
    return parser.parse_args()

#benutzt, wenn der Server gar nicht erreichbar ist.
def die(msg: str, code: int = 1):
    print(f"[FEHLER] {msg}", file=sys.stderr)
    sys.exit(code)

#Antwortet Fuseki auf http://localhost:3030
def check_fuseki_running(base_url: str, auth=None):
    try:
        r = requests.get(base_url, timeout=3, auth=auth)
        if r.status_code < 400:
            print(f"[OK] Fuseki antwortet unter {base_url}")
            return
        die(f"Fuseki antwortet mit Status {r.status_code}")
    except requests.exceptions.RequestException as e:
        die(f"Kann {base_url} nicht erreichen: {e}")

'''
Versucht, über die Admin-API (/$/datasets) zu schauen, ob dein Dataset schon existiert.
Wenn ja → „OK“.
Wenn nein → versucht es anzulegen.
Wenn die Admin-API geschützt ist → gibt nur einen Hinweis und macht weiter.
'''
def ensure_dataset_exists(base_url: str, dataset: str, auth=None):
    admin_url = f"{base_url}/$/datasets"
    try:
        r = requests.get(admin_url, timeout=3, auth=auth)
        if r.status_code != 200:
            print("[HINWEIS] Admin-API nicht erreichbar oder geschützt – überspringe Dataset-Anlage.")
            return

        datasets = r.json().get("datasets", [])
        names = [d.get("ds.name", "").lstrip("/") for d in datasets]
        if dataset in names:
            print(f"[OK] Dataset '{dataset}' existiert bereits.")
            return

        print(f"[INFO] Dataset '{dataset}' existiert nicht – versuche es anzulegen...")
        create_resp = requests.post(
            admin_url,
            data={"dbName": dataset, "dbType": "tdb2"},
            auth=auth,
        )
        if create_resp.status_code in (200, 201):
            print(f"[OK] Dataset '{dataset}' wurde angelegt.")
        else:
            print(f"[WARN] Konnte Dataset nicht anlegen (Status {create_resp.status_code}): "
                  f"{create_resp.text[:200]}")
    except requests.exceptions.RequestException as e:
        print(f"[WARN] Konnte Admin-API nicht verwenden: {e}")

'''
Öffnet TTL mit rdflib, führt SPARQL-Query aus, die nach aas:AssetAdministrationShell und deren Referable/idShort sucht.
'''
def get_idshort_from_ttl(path: str) -> Optional[str]:
    g = rdflib.Graph()
    g.parse(path, format="turtle")
    res = list(g.query(QUERY_IDSHORT))
    if res:
        return str(res[0][0])
    return None

'''
1. Versuch: Normale Fuseki-Weg
Macht ein POST auf
http://localhost:3030/<dataset>/data?graph=<dein-graph>
mit Content-Type: text/turtle und dem Dateiinhalt.
'''
def upload_via_graph_store(fuseki_data_url: str, file_path: str, graph_iri: str, auth=None) -> bool:
    with open(file_path, "rb") as f:
        r = requests.post(
            fuseki_data_url,
            params={"graph": graph_iri},
            headers={"Content-Type": "text/turtle"},
            data=f,
            auth=auth,
        )
    if r.status_code in (200, 201, 204):
        return True
    print(f"[INFO] Graph Store Upload fehlgeschlagen ({r.status_code}): {r.text[:150]}")
    return False


'''
2. Versuch!
Liest TTL nochmal ein, serialisiert sie zu N-Triples.
Baut ein SPARQL-Update:
'''
def upload_via_sparql_update(fuseki_update_url: str, file_path: str, graph_iri: str, auth=None) -> bool:
    g = rdflib.Graph()
    g.parse(file_path, format="turtle")
    ntriples = g.serialize(format="nt")
    if isinstance(ntriples, bytes):
        ntriples = ntriples.decode("utf-8")

    update = f"INSERT DATA {{ GRAPH <{graph_iri}> {{\n{ntriples}\n}} }}"
    r = requests.post(
        fuseki_update_url,
        data={"update": update},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        auth=auth,
    )
    if r.status_code not in (200, 204):
        print(f"[INFO] SPARQL UPDATE fehlgeschlagen ({r.status_code}): {r.text[:150]}")
    return r.status_code in (200, 204)



def upload_ttl_files_in_dir(
    data_dir: Union[str, Path],
    fuseki_base: Optional[str] = None,
    dataset: Optional[str] = None,
    auth: Optional[tuple] = None,
) -> Dict[Path, str]:
    """
    Lädt alle .ttl-Dateien aus 'data_dir' nach Fuseki hoch.

    Verwendet die bestehenden Funktionen:
      - check_fuseki_running
      - ensure_dataset_exists
      - get_idshort_from_ttl
      - upload_via_graph_store / upload_via_sparql_update

    Rückgabewert:
        dict {Pfad-zur-Datei -> graph_iri}
        (damit die UI z.B. die Init-Graph-URI kennt)
    """
    data_dir = Path(data_dir)
    fuseki_base = (fuseki_base or FUSEKI_BASE).rstrip("/")
    dataset = dataset or FUSEKI_DATASET
    if auth is None:
        # gleiche Defaults wie im CLI: admin/admin
        auth = ("admin", "admin")

    fuseki_data_url = f"{fuseki_base}/{dataset}/data"
    fuseki_update_url = f"{fuseki_base}/{dataset}/update"

    # 1) Fuseki prüfen / Dataset anlegen wie bisher
    check_fuseki_running(fuseki_base, auth=auth)
    ensure_dataset_exists(fuseki_base, dataset, auth=auth)

    if not data_dir.exists():
        print(f"[HINWEIS] Verzeichnis {data_dir} existiert nicht, nichts zu tun.")
        return {}

    mapping: Dict[Path, str] = {}

    for ttl_path in sorted(data_dir.glob("*.ttl")):
        try:
            idshort = get_idshort_from_ttl(str(ttl_path))
        except Exception as e:
            print(f"[WARN] Konnte idShort aus {ttl_path} nicht lesen ({e}), "
                  f"verwende Dateinamen.")
            idshort = None

        if not idshort:
            idshort = ttl_path.stem

        graph_iri = f"urn:aas:graph:{urllib.parse.quote(idshort)}"
        print(f"[INFO] Upload {ttl_path.name} → {graph_iri}")

        # 1. Versuch: Graph-Store HTTP
        ok = upload_via_graph_store(
            fuseki_data_url, str(ttl_path), graph_iri, auth=auth
        )
        # 2. Fallback: SPARQL-Update
        if not ok:
            ok = upload_via_sparql_update(
                fuseki_update_url, str(ttl_path), graph_iri, auth=auth
            )

        if ok:
            mapping[ttl_path] = graph_iri
        else:
            print(f"[ERROR] Konnte {ttl_path} nicht hochladen.")

    return mapping








###### KONSOLE #########


def main():
    args = parse_args()

    fuseki_base = args.fuseki.rstrip("/")
    dataset = args.dataset
    data_dir = args.dir

    auth = (args.user, args.password) if args.user and args.password else None
    print(f"AUTHENT '{auth}'")

    fuseki_data_url = f"{fuseki_base}/{dataset}/data"
    fuseki_update_url = f"{fuseki_base}/{dataset}/update"

    check_fuseki_running(fuseki_base, auth=auth)
    ensure_dataset_exists(fuseki_base, dataset, auth=auth)

    # Verzeichnis sicherstellen
    if not os.path.isdir(data_dir):
        print(f"[INFO] Verzeichnis '{data_dir}' existiert nicht – lege es an.")
        os.makedirs(data_dir, exist_ok=True)
        print(f"[HINWEIS] Bitte deine .ttl-Dateien nach '{data_dir}' legen und das Skript erneut ausführen.")
        sys.exit(0)

    ttl_files = [f for f in os.listdir(data_dir) if f.endswith(".ttl")]
    if not ttl_files:
        print(f"[HINWEIS] Keine .ttl-Dateien in '{data_dir}' gefunden.")
        sys.exit(0)

    for fname in ttl_files:
        file_path = os.path.join(data_dir, fname)
        print(f"\n--- Verarbeite {fname} ---")

        try:
            idshort = get_idshort_from_ttl(file_path)
            print(f"IDSHORT: ({idshort}) ")
        except Exception as e:  # Info: da gehts gar nicht rein, weil idshort = None manchmal!!!
            print(f"[WARN] Konnte idShort nicht auslesen ({e}) – verwende Dateinamen.")
            idshort = None #os.path.splitext(fname)[0]

        # Wenn kein idShort -> Dateiname nehmen
        if not idshort:
            idshort = os.path.splitext(fname)[0]    # ToDo: Idshort richtig / wichtig? --> Überlegung hier id nehmen oder so

        graph_iri = f"urn:aas:graph:{urllib.parse.quote(idshort)}"
        print(f"[INFO] Zielgraph: <{graph_iri}>")

        # 1. Versuch
        if upload_via_graph_store(fuseki_data_url, file_path, graph_iri, auth=auth):
            print(f"[OK] {fname} hochgeladen (Graph Store).")
            continue

        # 2. Versuch
        if upload_via_sparql_update(fuseki_update_url, file_path, graph_iri, auth=auth):
            print(f"[OK] {fname} hochgeladen (SPARQL UPDATE).")
        else:
            print(f"[FEHLER] {fname} konnte nicht hochgeladen werden.")


# Main wird gestartet, wenn das Skript direkt gestartet wird (, nicht bei Import in anderes Skript)
if __name__ == "__main__":
    main()