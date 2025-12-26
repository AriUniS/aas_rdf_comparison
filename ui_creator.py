from nicegui import ui, events
from random import randint
import logging
from pathlib import Path
from rdflib.plugins.sparql.parser import parseQuery
from typing import Optional

import os

# from data_processing.paths import (
#    RAW_JSON_INIT_DIR,
#    RAW_JSON_LIB_DIR,
#    TTL_INIT_DIR,
#    TTL_LIB_DIR,
# )
from data_processing.json_to_rdf import json_file_to_ttl_file
from filtered_output import embed_graphs  # , select_graphs

from backend_paths import FUSEKI_BASE, FUSEKI_DATASET, FUSEKI_ENDPOINT, TTL_INIT_DIR, TTL_LIB_DIR, RAW_JSON_INIT_DIR, \
    RAW_JSON_LIB_DIR
from similarity_measure import cosine_similarity
from upload import upload_ttl_files_in_dir, get_idshort_from_ttl
from services.pipeline import convert_jsons_to_ttl_and_upload
from services.sparql_filter import filter_graphs
from services.ranking import embed_and_rank_graphs
# from upload_ui import files_up

import urllib

import math


class UICreator():
    def __init__(self) -> None:
        self.current_step = 1
        self.sparql_query = ""
        self.init_json_files: list[Path] = []  # TODO: SPÄTER HINTERFRAGEN!!!
        self.lib_json_files: list[Path] = []

        self.filtered_graphs: list[str] = []  # Zustand nach SPARQL-Filter
        self.graph_embeddings: dict[str, list[float]] = {}  # Zustand nach

        self.init_graph_uri: Optional[str] = None  # Graph der Init-AAS
        self.init_ttl_path: Optional[Path] = None  # Pfad zur TTL der Init-AAS
        self.ranking_rows: list[dict] = []  # fertige Zeilen für die Tabelle in Step 3

    def show(self):
        """Entry-Point aus main.py"""
        self._render()

    # --------- STATE & NAVIGATION ---------

    def _next_step(self):
        if self.current_step == 2:
            self._vectorize_graphs()
        if self.current_step < 3:
            self.current_step += 1
            # z.B. Upload / SPARQL / Embeddings etc.
            self._render.refresh()

    def _prev_step(self):
        if self.current_step > 1:
            self.current_step -= 1
            # wenn man zurückgeht, werden spätere Schritte später
            # auf Basis der neuen Daten wieder neu berechnet
            self._render.refresh()

    def _clear_all(self):
        """Alles zurücksetzen, außer ggf. SPARQL wenn du willst."""
        # Wenn SPARQL auch gelöscht werden soll: self.sparql_query = ''
        self.current_step = 1
        self.sparql_query = ""
        self.init_json_files.clear()
        self.lib_json_files.clear()
        self.filtered_graphs.clear()
        self.graph_embeddings.clear()
        self.init_graph_uri = None
        self.init_ttl_path = None
        self.ranking_rows.clear()

        # Dateien in den Arbeitsordnern löschen
        self._clear_data_dirs()

        self._render.refresh()
        ui.notify('All local JSON/TTL files cleared and wizard reset.', color='green')

    def _clear_data_dirs(self) -> None:
        """Löscht alle JSON- und TTL-Dateien aus den Arbeitsordnern."""
        for directory in (RAW_JSON_INIT_DIR, RAW_JSON_LIB_DIR, TTL_INIT_DIR, TTL_LIB_DIR):
            try:
                if directory.exists():
                    for f in directory.iterdir():
                        if f.is_file():
                            try:
                                f.unlink()
                            except Exception as e:
                                logging.warning('Could not delete %s: %s', f, e)
            except Exception as e:
                logging.warning('Error accessing directory %s: %s', directory, e)

        # --------- UI RENDER ---------

    @ui.refreshable
    def _render(self):
        titles = {
            1: 'Choice of Data',
            2: 'Prefiltering with SPARQL: Define your constraints',
            3: 'Results after Filtering',
        }

        with (ui.column().classes('w-full gap-4')):
            # Step-Überschrift
            with ui.row().classes('items-baseline gap-2'):
                ui.label(f'{self.current_step}.').classes('text-2xl font-bold')
                ui.label(titles[self.current_step]).classes('text-2xl font-bold')

            # kleine Step-Anzeige + Navigation
            with ui.row().classes('w-full items-center justify-between mt-2'):
                back_button = ui.button('Back', icon='arrow_back',
                                        on_click=self._prev_step, )
                if self.current_step == 1:
                    back_button.disable()  # hier wird er deaktiviert

                ui.label(f'Step {self.current_step} / 3') \
                    .classes('text-sm text-gray-500')
                forward_button = ui.button('Forward', icon='arrow_forward',
                                           on_click=self._next_step)
                if self.current_step == 3:
                    forward_button.disable()  # hier wird er deaktiviert

            ui.separator()

            # Inhalt je Schritt
            if self.current_step == 1:
                self._step1()
            elif self.current_step == 2:
                self._step2()
            else:
                self._step3()

    # --------- STEP 1: Choice of Data ---------

    # -- Step 1: Upload Handling --
    async def _handle_init_upload(self, e: events.UploadEventArguments) -> None:
        """Single Init-AAS (JSON oder TTL) – wir speichern JSON im init-Ordner."""
        filename = e.file.name
        if not filename.lower().endswith('.json'):
            ui.notify(f'Skipping {filename}: keine JSON-Datei', color='orange')
            return

        target = RAW_JSON_INIT_DIR / filename
        # e.content ist ein Binary-Stream :contentReference[oaicite:2]{index=2}

        data: bytes = await e.file.read()  # <-- HIER: async read()
        target.write_bytes(data)

        self.init_json_files.append(target)

        ui.notify(f'Init JSON gespeichert: {filename}')

    async def _handle_lib_upload(self, e: events.UploadEventArguments) -> None:
        """Bibliotheks-AAS – alle JSON Dateien landen im lib-Ordner."""
        filename = e.file.name
        if not filename.lower().endswith('.json'):
            ui.notify(f'Skipping {filename}: keine JSON-Datei', color='orange')
            return

        target = RAW_JSON_LIB_DIR / filename

        data: bytes = await e.file.read()  # <-- HIER: async read()
        target.write_bytes(data)

        self.lib_json_files.append(target)

        ui.notify(f'Library JSON gespeichert: {filename}')

    # -- Step 1: Conversion Method --

    def _convert_uploaded_json_to_ttl(self) -> None:
        """Alle bisher hochgeladenen JSONs nach TTL konvertieren, abspeichern und nach Fuseki hochladen.
            UI-Schicht: ruft Service auf, setzt UI-State und zeigt Notifications.
            """

        # optional: Early exit, wenn noch nichts hochgeladen wurde
        if not self.init_json_files and not self.lib_json_files:
            ui.notify('No JSON files uploaded yet.', color='orange')
            return

        try:
            result = convert_jsons_to_ttl_and_upload(
                init_json_files=self.init_json_files,
                lib_json_files=self.lib_json_files,
                ttl_init_dir=TTL_INIT_DIR,
                ttl_lib_dir=TTL_LIB_DIR,
                fuseki_base=FUSEKI_BASE,
                fuseki_dataset=FUSEKI_DATASET,
                auth=("admin", "admin"),
            )
        except Exception as ex:
            logging.exception("Error in JSON→TTL→Upload pipeline")
            ui.notify(f'Error during conversion/upload: {ex}', color='red')
            return

        # --- UI-State aktualisieren (nur UI!) ---
        self.init_graph_uri = result.get("init_graph_uri")

        # init_ttl_path (optional im UI-State halten)
        # Service liefert hier bewusst keinen Path zurück; wenn du ihn brauchst,
        # kannst du ihn weiterhin wie vorher setzen oder im Service zurückgeben.
        # Für dein UI ist init_graph_uri wichtiger als init_ttl_path.

        # --- Notifications ---
        converted_files = result.get("converted_files", 0)

        if self.init_graph_uri:
            ui.notify(f'Init graph set to: {self.init_graph_uri}', color='green')
        else:
            ui.notify(
                'Warning: could not determine Init-AAS graph URI – '
                'similarity ranking will not work yet.',
                color='orange',
            )

        ui.notify(f'JSON → TTL + Upload abgeschlossen ({converted_files} Dateien)', color='green')

    # -- Step 1: Graphic Interface
    def _step1(self):
        with (ui.row().classes('w-full items-start gap-6')):
            # links: Reset-Button
            with ui.column().classes('items-center gap-2'):
                ui.button(icon='refresh', on_click=self._clear_all)
                ui.label('Clear input\nand reload') \
                    .classes('text-xs text-gray-600 text-center')

            # Mitte: Input-Felder & Preview
            with ui.column().classes('gap-3 grow'):
                with ui.row().classes('w-full gap-4'):
                    ui.upload(on_upload=self._handle_init_upload  # lambda e: ui.notify(f'Uploaded {e.file.name}')
                              , label='Incomplete AAS', max_files=1) \
                        .props('outlined dense').classes('w-full')
                    # ui.input(
                    #    label='Input: Init-AAS (JSON file OR TTL-file)',
                    # ).props('outlined dense').classes('w-full')

                    ui.upload(
                        multiple=True,
                        label='All files of a network (for now as multiple files)',
                        on_upload=self._handle_lib_upload,
                        # on_upload=handle_lib_upload,
                    ).props('outlined dense webkitdirectory').classes('w-full') \
                        # .label('Library Folder (JSON / TTL)')

                ui.label('Later: Connection to AAS-Server, without lokal files') \
                    .classes('text-xs italic text-gray-500')

            # rechts: Upload-Button
            with ui.column().classes('items-center gap-2'):
                ui.button(icon='cloud_upload',
                          # on_click=files_up())
                          on_click=self._convert_uploaded_json_to_ttl)  # \
                # .props('round unelevated color=grey-3')
                ui.label('Translate to TTL and').classes('text-xs text-gray-600 text-center')
                ui.label('upload to RDF store (Apache Jena)') \
                    .classes('text-xs text-gray-600 text-center')

    # --------- STEP 2: SPARQL Prefilter ---------

    # -- Step 2: SPARQL syntax check

    async def _check_sparql_syntax(self) -> None:
        """Prüft nur die SPARQL-Syntax, wie ein Editor."""
        query = (self.sparql_query or '').strip()
        if not query:
            ui.notify('Please enter SPARQL query', color='red')
            return

        try:
            # versucht nur zu PARSEN, keine Ausführung
            parseQuery(query)
        except Exception as e:
            ui.notify(f'Error: Syntax: {e}', color='red')
            return

        ui.notify('SPARQL syntax is OK', color='green')

    def _filter_graphs_by_sparql(self) -> None:
        """UI-Wrapper: ruft Service filter_graphs(...) auf und setzt UI-State."""
        try:
            graphs = filter_graphs(FUSEKI_ENDPOINT, self.sparql_query)
        except Exception as ex:
            logging.exception("Error while filtering graphs via service")
            ui.notify(f'Error while filtering graphs: {ex}', color='red')
            return

        self.filtered_graphs = graphs

        if not graphs:
            ui.notify('No graphs matched the SPARQL query (or dataset empty).', color='orange')
        else:
            # Wenn Query leer war, ist das "alle Graphen" – Service entscheidet das
            raw = (self.sparql_query or '').strip()
            if not raw:
                ui.notify(f'No SPARQL filter → using ALL {len(graphs)} graphs.', color='green')
            else:
                ui.notify(f'Found {len(graphs)} graphs.', color='green')

    # -- Step 2: Vectorization
    def _vectorize_filtered_graphs(self) -> None:
        """Berechnet RDF2Vec-Embeddings für alle gefilterten Graphen (Step 3)."""
        if not self.filtered_graphs:
            ui.notify('No graphs filtered yet. Run SPARQL filter first.', color='orange')
            return

        if not self.init_graph_uri:
            ui.notify('No init graph selected/uploaded yet (init_graph_uri missing).', color='orange')
            return

        try:
            result = embed_and_rank_graphs(
                fuseki_endpoint=FUSEKI_ENDPOINT,
                graphs=self.filtered_graphs,
                init_graph_uri=self.init_graph_uri,
            )
        except Exception as ex:
            logging.exception("Error while embedding/ranking graphs via service")
            ui.notify(f'Error while embedding/ranking graphs: {ex}', color='red')
            return

            # UI-State setzen: nutze einen klaren State-Container im UI
        # self.similarity_ranking = result["ranking"]

        # ui.notify(f'Ranking computed for {result["count"]} graphs.', color='green')
        rows = []
        for i, r in enumerate(result["ranking"], start=1):
            rows.append({
                "rank": f"#{i}",
                "graph": r["graph_uri"],
                "score": f"{r['similarity']:.4f}",
                # optional: falls du später Sortierung brauchst:
                # "score_float": r["similarity"],
            })

        self.ranking_rows = rows
        ui.notify(f'Ranking computed for {len(rows)} graphs.', color='green')

        # Damit Step 3 die Tabelle neu zeichnet:
        self._render.refresh()

    def _vectorize_graphs(self) -> None:
        print("HERE")

        self._filter_graphs_by_sparql()
        print(self.filtered_graphs)
        self._vectorize_filtered_graphs()

    # -- Step 2: Graphic Interface
    def _step2(self):
        with (ui.row().classes('w-full items-start gap-6')):
            with ui.column().classes('flex-1 gap-4'):
                # obere Info-Leiste
                with ui.row().classes('w-full gap-4 items-stretch'):
                    ui.label('Later: Information-Window: Number of Graphs etc',
                             ).props('outlined dense').classes('w-full')

                ui.card().classes('min-w-[260px]')

                # SPARQL Eingabe

                ui.label('Enter SPARQL query here:') \
                    .classes('text-base font-semibold')
                ui.label('For filtering out AAS, please ' \
                         'consider the AAS a graph and write: SELECT DISTINCT ?g...') \
                    .classes('text-xs italic text-gray-500')
                # ToDo: Bitte Das hier intuitiver / Auffangszenarien
                # ToDo: Sollte unabhängig von Variable funktionieren, vielleicht mit "Vorlage"

                ui.textarea(placeholder='SELECT DISTINCT ?g ' \
                                        '\n'
                                        '...') \
                    .bind_value(self, 'sparql_query') \
                    .props('outlined autogrow') \
                    .classes('w-full ') \
                    .on('blur', self._check_sparql_syntax)

                ui.label('[first, this can be a text field. '
                         'Later: MAYBE SPARQL field implementation like in Apache Jena]') \
                    .classes('text-xs italic text-gray-500')

                ui.label('Later: Output as a table or JSON like in Apache Jena',
                         ).props('outlined').classes('w-full mt-4')

                # blaue Info-Box rechts
                with ui.card().classes('bg-blue-500 text-white min-w-[260px]'):
                    ui.label('Idea: Maybe put Apache Jena interface') \
                        .classes('text-sm font-semibold')

                ui.label('Filter graphs by SPARQL').classes(
                    'text-xs text-gray-600 text-center'
                )
                ui.label('(click to update filtered graphs)').classes(
                    'text-xs text-gray-500 text-center'
                )

    # --------- STEP 3: Results after Filtering ---------

    def _step3(self):
        with ui.column().classes('w-full gap-4'):
            # ui.input(label='Initialized AAS: [name/ID]') \
            #   .props('outlined dense').classes('w-full')

            ui.input(
                label='Initialized AAS: [name/ID]',
            ).bind_value(self, 'init_graph_uri') \
                .props('outlined dense readonly') \
                .classes('w-full')

            ui.label('Ranking').classes('text-base font-semibold mt-2')

            # Tabelle als PLATZHALTER: PLS CHANGE --> ToDo!!!
            columns = [
                {'name': 'rank', 'label': 'Similarity Rank', 'field': 'rank'},
                {'name': 'graph', 'label': 'AAS / graph name', 'field': 'graph'},
                {'name': 'score', 'label': 'score', 'field': 'score'},
            ]
            #   rows = [
            #      {'rank': '#1', 'graph': 'Example-Graph-A', 'score': '0.91'},
            #     {'rank': '#2', 'graph': 'Example-Graph-B', 'score': '0.87'},
            # ]

            rows = self.ranking_rows or []
            #          for i, g in enumerate(self.graph_embeddings.keys(), start=1):
            #             rows.append({
            #                'rank': f'#{i}',
            #                'graph': g,
            #                'score': '',
            #            })

            ui.table(columns=columns, rows=rows) \
                .classes('w-full')

            ui.label('Visualization').classes('text-base font-semibold mt-4')

            ui.textarea(
                '[Later: Either visualization of every vector OR\n'
                'comparison 1 by 1 (Init AAS <-> Bib-AAS) by clicking on Bib-AAS]'
            ).props('outlined autogrow').classes('w-full min-h-[160px]')

            with ui.row().classes('w-full justify-between items-center mt-4'):
                # blaue Info-Box links
                with ui.card().classes('bg-blue-500 text-white'):
                    ui.label('Yet to be defined: Best visualization technique') \
                        .classes('text-sm')

                # Reset rechts
                with ui.column().classes('items-center gap-2'):
                    ui.button(icon='refresh', on_click=self._clear_all)
                    ui.label('Clear everything and\nreturn to page 1') \
                        .classes('text-xs text-gray-600 text-center')


'''
     def _convert_uploaded_json_to_ttl_OLD(self) -> None:
            """Alle bisher hochgeladenen JSONs nach TTL konvertieren und abspeichern."""
            converted = 0

            # Init
            self.init_ttl_path = None
            for json_path in self.init_json_files:
                ttl_path = TTL_INIT_DIR / (json_path.stem + '.ttl')
                json_file_to_ttl_file(json_path, ttl_path)
                self.init_ttl_path = ttl_path
                converted += 1

            # Lib
            for json_path in self.lib_json_files:
                ttl_path = TTL_LIB_DIR / (json_path.stem + '.ttl')
                json_file_to_ttl_file(json_path, ttl_path)
                converted += 1

            mapping_init = upload_ttl_files_in_dir(
                TTL_INIT_DIR,
                fuseki_base=FUSEKI_BASE,
                dataset=FUSEKI_DATASET,
                auth=("admin", "admin"),  # wie in upload.py-Defaults
            )
            mapping_lib = upload_ttl_files_in_dir(
                TTL_LIB_DIR,
                fuseki_base=FUSEKI_BASE,
                dataset=FUSEKI_DATASET,
                auth=("admin", "admin"),
            )

            # --- Init-Graph-URI merken ---
            self.init_graph_uri = None
            if self.init_ttl_path is not None:
                # mapping_init ist {Path -> graph_iri}
                for path, graph_iri in mapping_init.items():
                    if os.path.abspath(path) == os.path.abspath(self.init_ttl_path):
                        self.init_graph_uri = graph_iri
                        break

            if self.init_graph_uri:
                ui.notify(f'Init graph set to: {self.init_graph_uri}', color='green')
            else:
                ui.notify(
                    'Warning: could not determine Init-AAS graph URI – '
                    'similarity ranking will not work yet.',
                    color='orange',
                )

            ui.notify(f'JSON → TTL + Upload abgeschlossen ({converted} Dateien)', color='green')


        def _filter_graphs_by_sparql(self) -> None:
        """Wird in Step 2 vom Button aufgerufen: filtert Graphen per SPARQL, wenn vorhanden."""

        raw = (self.sparql_query or '').strip()

        # FALL 1: nichts eingegeben -> kein Filter, alle Graphen holen
        if not raw:

            default_query = """
                SELECT DISTINCT ?g WHERE {
                  GRAPH ?g {  }
                }
            """                     # ?s ?p ?o
            try:
                graphs=select_graphs(FUSEKI_ENDPOINT, default_query)
                ui.notify(f'SUCCESS FALL 1')

            except Exception as ex:
                logging.exception("Error while filtering graphs (no-filter mode)")
                ui.notify(f'Error while listing all graphs: {ex}', color='red')
                return

            print(f'GRAPHS:{graphs}')
            self.filtered_graphs = graphs
            if not graphs:
                ui.notify('No graphs found in dataset.', color='orange')
            else:
                ui.notify(f'No SPARQL filter → using ALL {len(graphs)} graphs.', color='green')
            return

        # FALL 2: es gibt eine Query -> normaler Syntaxcheck + Filter
        else:
            try:
                parseQuery(raw)
            except Exception as e:
                ui.notify(f'Error: Syntax: {e}', color='red')
                return

            try:
                graphs = select_graphs(FUSEKI_ENDPOINT, raw)
            except Exception as ex:
                logging.exception("Error while filtering graphs")
                ui.notify(f'Error while filtering graphs: {ex}', color='red')
                return

            self.filtered_graphs = graphs

            if not graphs:
                ui.notify('No graphs matched the SPARQL query.', color='orange')
            else:
                ui.notify(f'Found {len(graphs)} graphs.', color='green')




        def _vectorize_filtered_graphs(self) -> None:
        """Berechnet RDF2Vec-Embeddings für alle gefilterten Graphen (Step 3)."""
        if not self.filtered_graphs:
            ui.notify(
                'No filtered graphs yet – run the SPARQL filter first.',
                color='orange',
            )
            self.ranking_rows = []
            return

        try:
            all_embeddings, graph_vectors = embed_graphs(
                FUSEKI_ENDPOINT,
                self.filtered_graphs,
            )
            print("HERE1")

        except Exception as ex:
            logging.exception("Error during vectorization")
            ui.notify(f'Error during vectorization: {ex}', color='red')
            self.ranking_rows = []
            return

        self.graph_embeddings = graph_vectors

        # --- Similarity-Ranking nur, wenn ein Init-Graph gesetzt ist ---
        if not self.init_graph_uri:
            ui.notify(
                'No Init-AAS graph set (init_graph_uri) – cannot compute similarity.',
                color='orange',
            )
            self.ranking_rows = []
            return

        if self.init_graph_uri not in graph_vectors:
            ui.notify(
                f'Init graph {self.init_graph_uri} not among embedded graphs – '
                'check SPARQL filter or dataset.',
                color='orange',
            )
            self.ranking_rows = []
            return

        init_vec = graph_vectors[self.init_graph_uri]

        # Similarity zu allen anderen Graphen berechnen
        rows: list[dict] = []
        for uri, vec in graph_vectors.items():
            if uri == self.init_graph_uri:
                continue  # sich selbst nicht vergleichen

            sim = cosine_similarity(init_vec, vec)
            if math.isnan(sim):
                continue

            rows.append({
                'graph': uri,
                'score_float': sim,  # für Sortierung
                'score': f'{sim:.4f}',  # für Anzeige
            })

        # nach Similarity absteigend sortieren (höchste zuerst)
        rows.sort(key=lambda r: r['score_float'], reverse=True)

        # Ranking-Nummern hinzufügen
        for i, row in enumerate(rows, start=1):
            row['rank'] = f'#{i}'

        self.ranking_rows = rows
        ui.notify(f'Computed similarity for {len(rows)} graphs.')

'''