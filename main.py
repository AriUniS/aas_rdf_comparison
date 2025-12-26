from nicegui import ui, app
from pydantic import BaseModel

from ui_creator import UICreator
from interface.frame import frame

from rdf2vec_application import embed_ttl_graph, graph_embedding_from_nodes  # :contentReference[oaicite:3]{index=3}
from upload import upload_ttl_files_in_dir

from services.sparql_filter import filter_graphs
from services.ranking import embed_and_rank_graphs

from backend_paths import FUSEKI_ENDPOINT
from typing import Optional


uicreator = UICreator()

@ui.page('/')
async def main_page():
    with frame("AAS-graph-comparison"):
        uicreator.show()
        #uicreator.aas_editor()

ui.run(fastapi_docs=True)
class EmbedRequest(BaseModel):
    ttl: str

class UploadDirRequest(BaseModel):
    directory: str

class GraphFilterRequest(BaseModel):
    sparql: Optional[str]=None

class RankGraphsRequest(BaseModel):
    graphs: list[str]
    init_graph_uri: str


@app.post("/api/embed/nodes")
async def api_embed_nodes(req: EmbedRequest):
    return embed_ttl_graph(req.ttl)

@app.post("/api/embed/graph")
async def api_embed_graph(req: EmbedRequest):
    node_vecs = embed_ttl_graph(req.ttl)
    graph_vec = graph_embedding_from_nodes(node_vecs)
    return {
        "node_embeddings": node_vecs,
        "graph_embedding": graph_embedding_from_nodes(node_vecs),
    }

@app.post("/api/fuseki/upload_dir")
async def api_upload_dir(req: UploadDirRequest):
    mapping = upload_ttl_files_in_dir(req.directory)
    # Path -> str serialisieren:
    return {str(k): v for k, v in mapping.items()}


@app.post("/api/graphs/filter")
async def api_filter_graphs(req: GraphFilterRequest):
    graphs = filter_graphs(FUSEKI_ENDPOINT, req.sparql)
    return {"graphs": graphs, "count": len(graphs)}

@app.post("/api/graphs/rank")
async def api_rank_graphs(req: RankGraphsRequest):
    return embed_and_rank_graphs(
        fuseki_endpoint=FUSEKI_ENDPOINT,
        graphs=req.graphs,
        init_graph_uri=req.init_graph_uri,
    )

