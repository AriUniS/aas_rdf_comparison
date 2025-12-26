from pathlib import Path
import os
from typing import List, Dict, Any, Optional

from data_processing.json_to_rdf import json_file_to_ttl_file
from upload import upload_ttl_files_in_dir


def convert_jsons_to_ttl_and_upload(
    init_json_files: list[Path],
    lib_json_files: list[Path],
    ttl_init_dir: Path,
    ttl_lib_dir: Path,
    fuseki_base: str,
    fuseki_dataset: str,
    auth: tuple[str, str] = ("admin", "admin"),
):
    converted = 0
    init_ttl_path: Optional[Path] = None

    for json_path in init_json_files:
        ttl_path = ttl_init_dir / (json_path.stem + ".ttl")
        json_file_to_ttl_file(json_path, ttl_path)
        init_ttl_path = ttl_path
        converted += 1

    for json_path in lib_json_files:
        ttl_path = ttl_lib_dir / (json_path.stem + ".ttl")
        json_file_to_ttl_file(json_path, ttl_path)
        converted += 1

    mapping_init = upload_ttl_files_in_dir(ttl_init_dir, fuseki_base=fuseki_base, dataset=fuseki_dataset, auth=auth)
    mapping_lib  = upload_ttl_files_in_dir(ttl_lib_dir,  fuseki_base=fuseki_base, dataset=fuseki_dataset, auth=auth)

    init_graph_uri = None
    if init_ttl_path is not None:
        for path, graph_iri in mapping_init.items():
            if os.path.abspath(path) == os.path.abspath(init_ttl_path):
                init_graph_uri = graph_iri
                break

    return {
        "converted_files": converted,
        "mapping_init": {str(k): v for k, v in mapping_init.items()},
        "mapping_lib":  {str(k): v for k, v in mapping_lib.items()},
        "init_graph_uri": init_graph_uri,
    }