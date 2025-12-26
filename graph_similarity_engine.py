# graph_similarity_engine.py
from __future__ import annotations

import os
import re
import math
import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import XSD

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

import requests
from urllib.parse import quote


from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

# Optional for method A
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


LIT = Namespace("urn:litfeat:")


# -----------------------------
# Fuseki download (robust, nutzt /data?graph= oder /get?graph=)
# -----------------------------
def _try_get(endpoint: str, graph_uri: str, path: str, fmt: str, auth=None):
    base = endpoint.rstrip("/")
    url = f"{base}/{path}?graph={quote(graph_uri, safe='')}"
    return requests.get(url, headers={"Accept": fmt}, timeout=60, auth=auth)

def download_graph_ttl(endpoint: str, graph_uri: str, auth=None) -> str:
    # 1) /data
    r = _try_get(endpoint, graph_uri, "data", "text/turtle", auth=auth)
    if r.status_code == 200:
        return r.text
    # 2) /get
    r2 = _try_get(endpoint, graph_uri, "get", "text/turtle", auth=auth)
    if r2.status_code == 200:
        return r2.text
    raise requests.HTTPError(
        f"Konnte Graph {graph_uri} nicht laden. /data={r.status_code}, /get={r2.status_code}"
    )

# -----------------------------
# Helpers
# -----------------------------
def _hash_id(*parts: str, n: int = 16) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:n]


def _num_bin(x: float) -> str:
    ax = abs(x)
    if ax == 0:
        return "NUM_0"
    exp = int(math.floor(math.log10(ax)))
    exp = max(-6, min(6, exp))
    return f"NUM_1e{exp}"


def _len_bin(n: int) -> str:
    if n <= 5: return "LEN_1_5"
    if n <= 20: return "LEN_6_20"
    if n <= 100: return "LEN_21_100"
    return "LEN_101_PLUS"


def _text_shape(s: str) -> str:
    s = s.strip()
    if re.match(r"^https?://", s): return "TEXT_URL"
    if "@" in s and "." in s: return "TEXT_EMAIL"
    if re.match(r"^\+?\d[\d\s\-]{5,}$", s): return "TEXT_PHONE"
    if re.match(r"^\d+(\.\d+)?$", s): return "TEXT_NUMERIC_STRING"
    if re.match(r"^[A-Za-z0-9_\-]+$", s): return "TEXT_ALPHANUM"
    return "TEXT_FREE"


def _try_parse_date(lex: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(lex.replace("Z", "+00:00"))
    except Exception:
        return None


def _safe_float(lex: str) -> Optional[float]:
    try:
        return float(lex)
    except Exception:
        return None


def _duration_to_seconds_iso8601(lex: str) -> Optional[float]:
    """
    Sehr robuster, einfacher Parser für häufige ISO-8601 Durations.
    Unterstützt PnDTnHnMnS / PTnHnMnS / PnD usw.
    (Nicht vollständig für Monate/Jahre, weil variabel.)
    """
    # Example: "PT15M", "P2DT3H4M", "PT0.5S"
    m = re.match(
        r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?"
        r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$",
        lex.strip()
    )
    if not m:
        return None
    days = float(m.group("days")) if m.group("days") else 0.0
    hours = float(m.group("hours")) if m.group("hours") else 0.0
    minutes = float(m.group("minutes")) if m.group("minutes") else 0.0
    seconds = float(m.group("seconds")) if m.group("seconds") else 0.0
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, 0) or np.allclose(b, 0):
        return float("nan")
    return 1.0 - cosine(a, b)


# -----------------------------
# (1) Literal augmentation for RDF2Vec
# -----------------------------
def augment_literals(g: Graph) -> Graph:
    """
    Originaltriples bleiben erhalten.
    Zusätzlich: (s,p,litNode) + litNode-Features (typisiert, endlich).
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

        # extra edge, original stays:
        g2.add((s, p, lit_node))

        kind = "OTHER"

        if o.datatype in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                          XSD.decimal, XSD.double, XSD.float):
            kind = "NUM"
            x = _safe_float(lex)
            if x is not None:
                g2.add((lit_node, LIT.numBin, URIRef(LIT[_num_bin(x)])))

        elif o.datatype in (XSD.date, XSD.dateTime):
            kind = "DATE"
            dt = _try_parse_date(lex)
            if dt:
                g2.add((lit_node, LIT.year, URIRef(LIT[f"Y_{dt.year}"])))
                g2.add((lit_node, LIT.month, URIRef(LIT[f"YM_{dt.year}_{dt.month:02d}"])))

        elif o.datatype == XSD.duration:
            kind = "DURATION"
            g2.add((lit_node, LIT.durShape, URIRef(LIT["DUR_ISO8601"])))

        else:
            if lang or (o.datatype in (None, XSD.string)):
                kind = "TEXT"
                g2.add((lit_node, LIT.textShape, URIRef(LIT[_text_shape(lex)])))
                g2.add((lit_node, LIT.lenBin, URIRef(LIT[_len_bin(len(lex))])))
                if lang:
                    g2.add((lit_node, LIT.lang, URIRef(LIT[f"LANG_{lang}"])))

        g2.add((lit_node, LIT.kind, URIRef(LIT[kind])))
        if dt_uri:
            g2.add((lit_node, LIT.datatype, URIRef(LIT[_hash_id(dt_uri, n=10)])))

    return g2


def _entities_for_graph(g: Graph) -> List[str]:
    """
    Knoten, die wir später poolen.
    Prädikate nicht explizit poolen (beeinflussen aber Kontexte/W2V).
    """
    ents = set()
    for s in g.subjects():
        if isinstance(s, URIRef):
            ents.add(str(s))
    for o in g.objects():
        if isinstance(o, URIRef):
            ents.add(str(o))
    return sorted(ents)


# -----------------------------
# (2) Train ONE RDF2Vec model for ALL graphs
# -----------------------------
@dataclass
class RDF2VecConfig:
    vector_size: int = 100
    max_depth: int = 10
    max_walks: int = 10
    epochs: int = 5
    seed: int = 42


def train_rdf2vec_joint(ttl_by_uri: Dict[str, str], cfg: RDF2VecConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """
    Returns:
      emb_global: {entity_uri: vector}
      entities_by_graph: {graph_uri: [entity_uri,...]}
    """
    np.random.seed(cfg.seed)

    parsed_aug: Dict[str, Graph] = {}
    entities_by_graph: Dict[str, List[str]] = {}

    for gu, ttl in ttl_by_uri.items():
        g = Graph()
        g.parse(data=ttl, format="turtle")
        g_aug = augment_literals(g)
        parsed_aug[gu] = g_aug
        entities_by_graph[gu] = _entities_for_graph(g_aug)

    g_all = Graph()
    for g in parsed_aug.values():
        for t in g:
            g_all.add(t)

    entities_all = sorted(set(e for ents in entities_by_graph.values() for e in ents))
    if not entities_all:
        return {}, entities_by_graph

    with tempfile.NamedTemporaryFile("w+", suffix=".ttl", delete=False, encoding="utf-8") as tmp:
        tmp.write(g_all.serialize(format="turtle"))
        tmp_path = tmp.name

    try:
        kg = KG(location=tmp_path, fmt="turtle", is_remote=False)

        transformer = RDF2VecTransformer(
            Word2Vec(vector_size=cfg.vector_size, sg=1, epochs=cfg.epochs),
            walkers=[RandomWalker(cfg.max_depth, cfg.max_walks, with_reverse=True)],
        )

        mat, _ = transformer.fit_transform(kg, entities_all)
        emb_global = {entities_all[i]: np.array(mat[i], dtype=float) for i in range(len(entities_all))}
        return emb_global, entities_by_graph

    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def pool_graph_vector(emb_global: Dict[str, np.ndarray], entities: List[str]) -> np.ndarray:
    vecs = [emb_global[e] for e in entities if e in emb_global]
    if not vecs:
        return np.zeros((1,), dtype=float)
    return np.mean(np.vstack(vecs), axis=0)


# -----------------------------
# (A) Literal VALUE vectors: TF-IDF + numeric/date/duration stats
# -----------------------------
@dataclass
class LiteralAConfig:
    max_tfidf_features: int = 5000
    # numeric histogram bins (approx closeness)
    num_hist_bins: int = 20
    # date histogram bins by year (clipped)
    year_min: int = 1990
    year_max: int = 2035


class LiteralValueEncoderA:
    """
    Fits on a set of graphs (jointly) to ensure all graphs are in the same feature space:
      - TF-IDF vocabulary/IDF from all texts
      - scaling from all numeric/date/duration features
    Produces per-graph fixed-size vectors.
    """

    def __init__(self, cfg: LiteralAConfig):
        self.cfg = cfg
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=cfg.max_tfidf_features,
        )
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self._fitted = False

    def _extract_literals(self, g: Graph) -> Tuple[List[str], List[float], List[int], List[float]]:
        texts: List[str] = []
        nums: List[float] = []
        years: List[int] = []
        durs: List[float] = []

        for _, _, o in g.triples((None, None, None)):
            if not isinstance(o, Literal):
                continue

            lex = str(o)
            dt = o.datatype

            if dt in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                      XSD.decimal, XSD.double, XSD.float):
                x = _safe_float(lex)
                if x is not None:
                    nums.append(x)

            elif dt in (XSD.date, XSD.dateTime):
                d = _try_parse_date(lex)
                if d is not None:
                    years.append(d.year)

            elif dt == XSD.duration:
                sec = _duration_to_seconds_iso8601(lex)
                if sec is not None:
                    durs.append(sec)

            else:
                # treat as text if string-ish or language-tag
                if o.language or dt in (None, XSD.string):
                    if lex.strip():
                        texts.append(lex.strip())

        return texts, nums, years, durs

    def _numeric_feature_block(self, nums: List[float], years: List[int], durs: List[float]) -> np.ndarray:
        # Numeric summary (captures metric closeness partially)
        def safe_stats(xs: List[float]) -> List[float]:
            if not xs:
                return [0.0] * 6
            arr = np.array(xs, dtype=float)
            return [
                float(arr.mean()),
                float(arr.std()),
                float(np.median(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(len(arr)),
            ]

        num_stats = safe_stats(nums)

        # Histogram for numbers (log scale stabilizes)
        if nums:
            arr = np.array(nums, dtype=float)
            arr = np.sign(arr) * np.log1p(np.abs(arr))
            hist, _ = np.histogram(arr, bins=self.cfg.num_hist_bins)
            num_hist = hist.astype(float).tolist()
        else:
            num_hist = [0.0] * self.cfg.num_hist_bins

        # Years histogram
        ybins = list(range(self.cfg.year_min, self.cfg.year_max + 2))  # inclusive
        if years:
            yarr = np.clip(np.array(years, dtype=int), self.cfg.year_min, self.cfg.year_max)
            yhist, _ = np.histogram(yarr, bins=ybins)
            year_hist = yhist.astype(float).tolist()
        else:
            year_hist = [0.0] * (len(ybins) - 1)

        dur_stats = safe_stats(durs)

        return np.array(num_stats + num_hist + year_hist + dur_stats, dtype=float)

    def fit(self, ttl_by_uri: Dict[str, str]) -> "LiteralValueEncoderA":
        # Build corpora per graph
        texts_per_graph: List[str] = []
        numeric_blocks: List[np.ndarray] = []

        for _, ttl in ttl_by_uri.items():
            g = Graph()
            g.parse(data=ttl, format="turtle")
            texts, nums, years, durs = self._extract_literals(g)

            texts_per_graph.append(" ".join(texts))
            numeric_blocks.append(self._numeric_feature_block(nums, years, durs))

        # TF-IDF fit (joint)
        self.tfidf.fit(texts_per_graph)

        # Fit scaler on numeric blocks (joint)
        num_mat = np.vstack(numeric_blocks) if numeric_blocks else np.zeros((1, 1), dtype=float)
        self.scaler.fit(num_mat)

        self._fitted = True
        return self

    def transform_one(self, ttl: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LiteralValueEncoderA not fitted. Call fit(...) first.")

        g = Graph()
        g.parse(data=ttl, format="turtle")
        texts, nums, years, durs = self._extract_literals(g)

        text_doc = " ".join(texts)
        tfidf_vec = self.tfidf.transform([text_doc]).toarray().astype(float).reshape(-1)

        num_block = self._numeric_feature_block(nums, years, durs).reshape(1, -1)
        num_scaled = self.scaler.transform(num_block).reshape(-1)

        return np.concatenate([tfidf_vec, num_scaled], axis=0)

    def transform_all(self, ttl_by_uri: Dict[str, str]) -> Dict[str, np.ndarray]:
        return {gu: self.transform_one(ttl) for gu, ttl in ttl_by_uri.items()}


# -----------------------------
# (B) Literal VALUE vectors: Sentence-Embeddings for text + numeric/date/duration stats
# -----------------------------
@dataclass
class LiteralBConfig:
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_hist_bins: int = 20
    year_min: int = 1990
    year_max: int = 2035


class LiteralValueEncoderB:
    """
    Uses a pretrained sentence embedding model for text literals.
    Numeric/date/duration stats are scaled jointly.
    Requires: pip install sentence-transformers
    """

    def __init__(self, cfg: LiteralBConfig):
        self.cfg = cfg
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self._fitted = False
        self._model = None

    def _load_model(self):
        if self._model is None:
            #from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg.sbert_model_name)

    def _extract_literals(self, g: Graph) -> Tuple[List[str], List[float], List[int], List[float]]:
        texts: List[str] = []
        nums: List[float] = []
        years: List[int] = []
        durs: List[float] = []

        for _, _, o in g.triples((None, None, None)):
            if not isinstance(o, Literal):
                continue

            lex = str(o)
            dt = o.datatype

            if dt in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                      XSD.decimal, XSD.double, XSD.float):
                x = _safe_float(lex)
                if x is not None:
                    nums.append(x)

            elif dt in (XSD.date, XSD.dateTime):
                d = _try_parse_date(lex)
                if d is not None:
                    years.append(d.year)

            elif dt == XSD.duration:
                sec = _duration_to_seconds_iso8601(lex)
                if sec is not None:
                    durs.append(sec)

            else:
                if o.language or dt in (None, XSD.string):
                    if lex.strip():
                        texts.append(lex.strip())

        return texts, nums, years, durs

    def _numeric_feature_block(self, nums: List[float], years: List[int], durs: List[float]) -> np.ndarray:
        def safe_stats(xs: List[float]) -> List[float]:
            if not xs:
                return [0.0] * 6
            arr = np.array(xs, dtype=float)
            return [
                float(arr.mean()),
                float(arr.std()),
                float(np.median(arr)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(len(arr)),
            ]

        num_stats = safe_stats(nums)

        if nums:
            arr = np.array(nums, dtype=float)
            arr = np.sign(arr) * np.log1p(np.abs(arr))
            hist, _ = np.histogram(arr, bins=self.cfg.num_hist_bins)
            num_hist = hist.astype(float).tolist()
        else:
            num_hist = [0.0] * self.cfg.num_hist_bins

        ybins = list(range(self.cfg.year_min, self.cfg.year_max + 2))
        if years:
            yarr = np.clip(np.array(years, dtype=int), self.cfg.year_min, self.cfg.year_max)
            yhist, _ = np.histogram(yarr, bins=ybins)
            year_hist = yhist.astype(float).tolist()
        else:
            year_hist = [0.0] * (len(ybins) - 1)

        dur_stats = safe_stats(durs)

        return np.array(num_stats + num_hist + year_hist + dur_stats, dtype=float)

    def fit(self, ttl_by_uri: Dict[str, str]) -> "LiteralValueEncoderB":
        self._load_model()

        numeric_blocks: List[np.ndarray] = []
        for _, ttl in ttl_by_uri.items():
            g = Graph()
            g.parse(data=ttl, format="turtle")
            _, nums, years, durs = self._extract_literals(g)
            numeric_blocks.append(self._numeric_feature_block(nums, years, durs))

        num_mat = np.vstack(numeric_blocks) if numeric_blocks else np.zeros((1, 1), dtype=float)
        self.scaler.fit(num_mat)

        self._fitted = True
        return self

    def transform_one(self, ttl: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LiteralValueEncoderB not fitted. Call fit(...) first.")
        self._load_model()

        g = Graph()
        g.parse(data=ttl, format="turtle")
        texts, nums, years, durs = self._extract_literals(g)

        # Sentence embeddings: mean pool over all text literals
        if texts:
            emb = self._model.encode(texts, normalize_embeddings=True)
            text_vec = np.mean(np.array(emb, dtype=float), axis=0)
        else:
            # need model dim for zeros:
            dim = self._model.get_sentence_embedding_dimension()
            text_vec = np.zeros((dim,), dtype=float)

        num_block = self._numeric_feature_block(nums, years, durs).reshape(1, -1)
        num_scaled = self.scaler.transform(num_block).reshape(-1)

        return np.concatenate([text_vec, num_scaled], axis=0)

    def transform_all(self, ttl_by_uri: Dict[str, str]) -> Dict[str, np.ndarray]:
        return {gu: self.transform_one(ttl) for gu, ttl in ttl_by_uri.items()}


# -----------------------------
# Final: build comparable graph vectors & similarities (A vs many)
# -----------------------------
@dataclass
class FusionConfig:
    rdf2vec_weight: float = 1.0
    literal_weight: float = 1.0
    normalize_parts: bool = True


def _l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def build_fused_vectors(
    ttl_by_uri: Dict[str, str],
    rdf_cfg: RDF2VecConfig,
    literal_encoder,  # A or B encoder, must have fit(...) and transform_all(...)
    fusion_cfg: FusionConfig
) -> Dict[str, np.ndarray]:
    # 1) Train one RDF2Vec space on all graphs
    emb_global, entities_by_graph = train_rdf2vec_joint(ttl_by_uri, rdf_cfg)

    # 2) Graph vectors from RDF2Vec (pooled)
    rdf_vecs: Dict[str, np.ndarray] = {}
    for gu, ents in entities_by_graph.items():
        rdf_vecs[gu] = pool_graph_vector(emb_global, ents)

    # 3) Literal value vectors (fit jointly => same space), Value encoders must be fit on the SAME set to keep one feature space
    literal_encoder.fit(ttl_by_uri)
    lit_vecs = literal_encoder.transform_all(ttl_by_uri)

    # 4) Fuse
    fused: Dict[str, np.ndarray] = {}
    for gu in ttl_by_uri.keys():
        rv = rdf_vecs.get(gu, None)
        lv = lit_vecs.get(gu, None)
        if rv is None or lv is None:
            continue

        if fusion_cfg.normalize_parts:
            rv = _l2norm(rv)
            lv = _l2norm(lv)

        fused_vec = np.concatenate([
            fusion_cfg.rdf2vec_weight * rv,
            fusion_cfg.literal_weight * lv
        ], axis=0)

        fused[gu] = _l2norm(fused_vec)  # normalize final for cosine stability

    return fused


def rank_against_anchor(anchor_uri: str, fused_vecs: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    if anchor_uri not in fused_vecs:
        raise ValueError(f"Anchor {anchor_uri} not in fused_vecs")

    a = fused_vecs[anchor_uri]
    scores = []
    for gu, v in fused_vecs.items():
        if gu == anchor_uri:
            continue
        scores.append((gu, cosine_sim(a, v)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores



# graph_similarity_engine.py
from __future__ import annotations

import os
import re
import math
import hashlib
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import XSD

import requests
from urllib.parse import quote

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker

from scipy.spatial.distance import cosine

# Method A deps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

LIT = Namespace("urn:litfeat:")


# -----------------------------
# Fuseki download (robust, nutzt /data?graph= oder /get?graph=)
# -----------------------------
def _try_get(endpoint: str, graph_uri: str, path: str, fmt: str, auth=None):
    base = endpoint.rstrip("/")
    url = f"{base}/{path}?graph={quote(graph_uri, safe='')}"
    return requests.get(url, headers={"Accept": fmt}, timeout=60, auth=auth)

def download_graph_ttl(endpoint: str, graph_uri: str, auth=None) -> str:
    # 1) /data
    r = _try_get(endpoint, graph_uri, "data", "text/turtle", auth=auth)
    if r.status_code == 200:
        return r.text
    # 2) /get
    r2 = _try_get(endpoint, graph_uri, "get", "text/turtle", auth=auth)
    if r2.status_code == 200:
        return r2.text
    raise requests.HTTPError(
        f"Konnte Graph {graph_uri} nicht laden. /data={r.status_code}, /get={r2.status_code}"
    )


# -----------------------------
# Literal augmentation for RDF2Vec (ohne Original zu entfernen)
# -----------------------------
def _hash_id(*parts: str, n: int = 16) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:n]

def _num_bin(x: float) -> str:
    ax = abs(x)
    if ax == 0:
        return "NUM_0"
    exp = int(math.floor(math.log10(ax)))
    exp = max(-6, min(6, exp))
    return f"NUM_1e{exp}"

def _len_bin(n: int) -> str:
    if n <= 5: return "LEN_1_5"
    if n <= 20: return "LEN_6_20"
    if n <= 100: return "LEN_21_100"
    return "LEN_101_PLUS"

def _text_shape(s: str) -> str:
    s = s.strip()
    if re.match(r"^https?://", s): return "TEXT_URL"
    if "@" in s and "." in s: return "TEXT_EMAIL"
    if re.match(r"^\+?\d[\d\s\-]{5,}$", s): return "TEXT_PHONE"
    if re.match(r"^\d+(\.\d+)?$", s): return "TEXT_NUMERIC_STRING"
    if re.match(r"^[A-Za-z0-9_\-]+$", s): return "TEXT_ALPHANUM"
    return "TEXT_FREE"

def _try_parse_date(lex: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(lex.replace("Z", "+00:00"))
    except Exception:
        return None

def _safe_float(lex: str) -> Optional[float]:
    try:
        return float(lex)
    except Exception:
        return None

def _duration_to_seconds_iso8601(lex: str) -> Optional[float]:
    # Simple ISO8601 duration parser for PnDTnHnMnS / PTnHnMnS (no months/years)
    m = re.match(
        r"^P(?:(?P<days>\d+(?:\.\d+)?)D)?"
        r"(?:T(?:(?P<hours>\d+(?:\.\d+)?)H)?(?:(?P<minutes>\d+(?:\.\d+)?)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?$",
        lex.strip()
    )
    if not m:
        return None
    days = float(m.group("days")) if m.group("days") else 0.0
    hours = float(m.group("hours")) if m.group("hours") else 0.0
    minutes = float(m.group("minutes")) if m.group("minutes") else 0.0
    seconds = float(m.group("seconds")) if m.group("seconds") else 0.0
    return days * 86400.0 + hours * 3600.0 + minutes * 60.0 + seconds

def augment_literals(g: Graph) -> Graph:
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

        # original triple remains; we ADD:
        g2.add((s, p, lit_node))

        kind = "OTHER"

        if o.datatype in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                          XSD.decimal, XSD.double, XSD.float):
            kind = "NUM"
            x = _safe_float(lex)
            if x is not None:
                g2.add((lit_node, LIT.numBin, URIRef(LIT[_num_bin(x)])))

        elif o.datatype in (XSD.date, XSD.dateTime):
            kind = "DATE"
            dt = _try_parse_date(lex)
            if dt:
                g2.add((lit_node, LIT.year, URIRef(LIT[f"Y_{dt.year}"])))
                g2.add((lit_node, LIT.month, URIRef(LIT[f"YM_{dt.year}_{dt.month:02d}"])))

        elif o.datatype == XSD.duration:
            kind = "DURATION"
            # keep coarse feature here; value semantics handled by value-encoder
            g2.add((lit_node, LIT.durShape, URIRef(LIT["DUR_ISO8601"])))

        else:
            if lang or (o.datatype in (None, XSD.string)):
                kind = "TEXT"
                g2.add((lit_node, LIT.textShape, URIRef(LIT[_text_shape(lex)])))
                g2.add((lit_node, LIT.lenBin, URIRef(LIT[_len_bin(len(lex))])))
                if lang:
                    g2.add((lit_node, LIT.lang, URIRef(LIT[f"LANG_{lang}"])))

        g2.add((lit_node, LIT.kind, URIRef(LIT[kind])))
        if dt_uri:
            g2.add((lit_node, LIT.datatype, URIRef(LIT[_hash_id(dt_uri, n=10)])))

    return g2

def _entities_for_graph(g: Graph) -> List[str]:
    ents = set()
    for s in g.subjects():
        if isinstance(s, URIRef):
            ents.add(str(s))
    for o in g.objects():
        if isinstance(o, URIRef):
            ents.add(str(o))
    return sorted(ents)


# -----------------------------
# Joint RDF2Vec training (one space)
# -----------------------------
@dataclass
class RDF2VecConfig:
    vector_size: int = 100
    max_depth: int = 10
    max_walks: int = 10
    epochs: int = 5
    seed: int = 42

def train_rdf2vec_joint(ttl_by_uri: Dict[str, str], cfg: RDF2VecConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    np.random.seed(cfg.seed)

    parsed_aug: Dict[str, Graph] = {}
    entities_by_graph: Dict[str, List[str]] = {}

    for gu, ttl in ttl_by_uri.items():
        g = Graph()
        g.parse(data=ttl, format="turtle")
        g_aug = augment_literals(g)
        parsed_aug[gu] = g_aug
        entities_by_graph[gu] = _entities_for_graph(g_aug)

    g_all = Graph()
    for g in parsed_aug.values():
        for t in g:
            g_all.add(t)

    entities_all = sorted(set(e for ents in entities_by_graph.values() for e in ents))
    if not entities_all:
        return {}, entities_by_graph

    with tempfile.NamedTemporaryFile("w+", suffix=".ttl", delete=False, encoding="utf-8") as tmp:
        tmp.write(g_all.serialize(format="turtle"))
        tmp_path = tmp.name

    try:
        kg = KG(location=tmp_path, fmt="turtle", is_remote=False)
        transformer = RDF2VecTransformer(
            Word2Vec(vector_size=cfg.vector_size, sg=1, epochs=cfg.epochs),
            walkers=[RandomWalker(cfg.max_depth, cfg.max_walks, with_reverse=True)],
        )
        mat, _ = transformer.fit_transform(kg, entities_all)
        emb_global = {entities_all[i]: np.array(mat[i], dtype=float) for i in range(len(entities_all))}
        return emb_global, entities_by_graph
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def pool_graph_vector(emb_global: Dict[str, np.ndarray], entities: List[str]) -> np.ndarray:
    vecs = [emb_global[e] for e in entities if e in emb_global]
    if not vecs:
        # keep deterministic dim if possible:
        if emb_global:
            dim = next(iter(emb_global.values())).shape[0]
            return np.zeros((dim,), dtype=float)
        return np.zeros((1,), dtype=float)
    return np.mean(np.vstack(vecs), axis=0)


# -----------------------------
# Literal VALUE encoders (A and B)
# -----------------------------
@dataclass
class LiteralAConfig:
    max_tfidf_features: int = 5000
    num_hist_bins: int = 20
    year_min: int = 1990
    year_max: int = 2035

class LiteralValueEncoderA:
    """
    Learns:
      - TF-IDF vocabulary+idf on all graphs (joint)
      - Scaling for numeric/date/duration stats (joint)
    """
    def __init__(self, cfg: LiteralAConfig):
        self.cfg = cfg
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
            max_features=cfg.max_tfidf_features,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def _extract_literals(self, g: Graph) -> Tuple[List[str], List[float], List[int], List[float]]:
        texts, nums, years, durs = [], [], [], []
        for _, _, o in g.triples((None, None, None)):
            if not isinstance(o, Literal):
                continue
            lex = str(o)
            dt = o.datatype
            if dt in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                      XSD.decimal, XSD.double, XSD.float):
                x = _safe_float(lex)
                if x is not None:
                    nums.append(x)
            elif dt in (XSD.date, XSD.dateTime):
                d = _try_parse_date(lex)
                if d is not None:
                    years.append(d.year)
            elif dt == XSD.duration:
                sec = _duration_to_seconds_iso8601(lex)
                if sec is not None:
                    durs.append(sec)
            else:
                if o.language or dt in (None, XSD.string):
                    if lex.strip():
                        texts.append(lex.strip())
        return texts, nums, years, durs

    def _numeric_block(self, nums: List[float], years: List[int], durs: List[float]) -> np.ndarray:
        def stats(xs: List[float]) -> List[float]:
            if not xs:
                return [0.0] * 6
            a = np.array(xs, dtype=float)
            return [a.mean(), a.std(), np.median(a), a.min(), a.max(), float(len(a))]

        num_stats = stats(nums)

        if nums:
            a = np.array(nums, dtype=float)
            a = np.sign(a) * np.log1p(np.abs(a))
            hist, _ = np.histogram(a, bins=self.cfg.num_hist_bins)
            num_hist = hist.astype(float).tolist()
        else:
            num_hist = [0.0] * self.cfg.num_hist_bins

        ybins = list(range(self.cfg.year_min, self.cfg.year_max + 2))
        if years:
            y = np.clip(np.array(years, dtype=int), self.cfg.year_min, self.cfg.year_max)
            yhist, _ = np.histogram(y, bins=ybins)
            year_hist = yhist.astype(float).tolist()
        else:
            year_hist = [0.0] * (len(ybins) - 1)

        dur_stats = stats(durs)

        return np.array(num_stats + num_hist + year_hist + dur_stats, dtype=float)

    def fit(self, ttl_by_uri: Dict[str, str]) -> "LiteralValueEncoderA":
        docs = []
        blocks = []
        for ttl in ttl_by_uri.values():
            g = Graph(); g.parse(data=ttl, format="turtle")
            texts, nums, years, durs = self._extract_literals(g)
            docs.append(" ".join(texts))
            blocks.append(self._numeric_block(nums, years, durs))
        self.tfidf.fit(docs)
        self.scaler.fit(np.vstack(blocks))
        self._fitted = True
        return self

    def transform_one(self, ttl: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LiteralValueEncoderA not fitted.")
        g = Graph(); g.parse(data=ttl, format="turtle")
        texts, nums, years, durs = self._extract_literals(g)
        doc = " ".join(texts)
        tf = self.tfidf.transform([doc]).toarray().reshape(-1)
        nb = self._numeric_block(nums, years, durs).reshape(1, -1)
        nb = self.scaler.transform(nb).reshape(-1)
        return np.concatenate([tf, nb], axis=0)

    def transform_all(self, ttl_by_uri: Dict[str, str]) -> Dict[str, np.ndarray]:
        return {gu: self.transform_one(ttl) for gu, ttl in ttl_by_uri.items()}


@dataclass
class LiteralBConfig:
    sbert_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_hist_bins: int = 20
    year_min: int = 1990
    year_max: int = 2035

class LiteralValueEncoderB:
    """
    Uses pretrained sentence-transformers for text literals (no training needed there),
    but still fits a scaler for numeric/date/duration blocks jointly.
    """
    def __init__(self, cfg: LiteralBConfig):
        self.cfg = cfg
        self.scaler = StandardScaler()
        self._fitted = False
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg.sbert_model_name)

    def _extract_literals(self, g: Graph) -> Tuple[List[str], List[float], List[int], List[float]]:
        # same as A
        texts, nums, years, durs = [], [], [], []
        for _, _, o in g.triples((None, None, None)):
            if not isinstance(o, Literal):
                continue
            lex = str(o)
            dt = o.datatype
            if dt in (XSD.integer, XSD.int, XSD.long, XSD.short, XSD.byte,
                      XSD.decimal, XSD.double, XSD.float):
                x = _safe_float(lex)
                if x is not None:
                    nums.append(x)
            elif dt in (XSD.date, XSD.dateTime):
                d = _try_parse_date(lex)
                if d is not None:
                    years.append(d.year)
            elif dt == XSD.duration:
                sec = _duration_to_seconds_iso8601(lex)
                if sec is not None:
                    durs.append(sec)
            else:
                if o.language or dt in (None, XSD.string):
                    if lex.strip():
                        texts.append(lex.strip())
        return texts, nums, years, durs

    def _numeric_block(self, nums: List[float], years: List[int], durs: List[float]) -> np.ndarray:
        # reuse logic from A
        def stats(xs: List[float]) -> List[float]:
            if not xs:
                return [0.0] * 6
            a = np.array(xs, dtype=float)
            return [a.mean(), a.std(), np.median(a), a.min(), a.max(), float(len(a))]

        num_stats = stats(nums)
        if nums:
            a = np.array(nums, dtype=float)
            a = np.sign(a) * np.log1p(np.abs(a))
            hist, _ = np.histogram(a, bins=self.cfg.num_hist_bins)
            num_hist = hist.astype(float).tolist()
        else:
            num_hist = [0.0] * self.cfg.num_hist_bins

        ybins = list(range(self.cfg.year_min, self.cfg.year_max + 2))
        if years:
            y = np.clip(np.array(years, dtype=int), self.cfg.year_min, self.cfg.year_max)
            yhist, _ = np.histogram(y, bins=ybins)
            year_hist = yhist.astype(float).tolist()
        else:
            year_hist = [0.0] * (len(ybins) - 1)

        dur_stats = stats(durs)
        return np.array(num_stats + num_hist + year_hist + dur_stats, dtype=float)

    def fit(self, ttl_by_uri: Dict[str, str]) -> "LiteralValueEncoderB":
        self._load_model()
        blocks = []
        for ttl in ttl_by_uri.values():
            g = Graph(); g.parse(data=ttl, format="turtle")
            _, nums, years, durs = self._extract_literals(g)
            blocks.append(self._numeric_block(nums, years, durs))
        self.scaler.fit(np.vstack(blocks))
        self._fitted = True
        return self

    def transform_one(self, ttl: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LiteralValueEncoderB not fitted.")
        self._load_model()

        g = Graph(); g.parse(data=ttl, format="turtle")
        texts, nums, years, durs = self._extract_literals(g)

        if texts:
            emb = self._model.encode(texts, normalize_embeddings=True)
            text_vec = np.mean(np.array(emb, dtype=float), axis=0)
        else:
            text_vec = np.zeros((self._model.get_sentence_embedding_dimension(),), dtype=float)

        nb = self._numeric_block(nums, years, durs).reshape(1, -1)
        nb = self.scaler.transform(nb).reshape(-1)

        return np.concatenate([text_vec, nb], axis=0)

    def transform_all(self, ttl_by_uri: Dict[str, str]) -> Dict[str, np.ndarray]:
        return {gu: self.transform_one(ttl) for gu, ttl in ttl_by_uri.items()}


# -----------------------------
# Fusion + Ranking
# -----------------------------
@dataclass
class FusionConfig:
    rdf2vec_weight: float = 1.0
    literal_weight: float = 1.0
    normalize_parts: bool = True

def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def build_fused_vectors(
    ttl_by_uri: Dict[str, str],
    rdf_cfg: RDF2VecConfig,
    literal_encoder: Any,   # LiteralValueEncoderA or B
    fusion_cfg: FusionConfig
) -> Dict[str, np.ndarray]:
    # 1) One RDF2Vec space for all
    emb_global, entities_by_graph = train_rdf2vec_joint(ttl_by_uri, rdf_cfg)

    rdf_vecs: Dict[str, np.ndarray] = {}
    for gu, ents in entities_by_graph.items():
        rdf_vecs[gu] = pool_graph_vector(emb_global, ents)

    # 2) Value encoders must be fit on the SAME set to keep one feature space
    literal_encoder.fit(ttl_by_uri)
    lit_vecs = literal_encoder.transform_all(ttl_by_uri)

    fused: Dict[str, np.ndarray] = {}
    for gu in ttl_by_uri.keys():
        rv = rdf_vecs.get(gu)
        lv = lit_vecs.get(gu)
        if rv is None or lv is None:
            continue

        if fusion_cfg.normalize_parts:
            rv = _l2(rv)
            lv = _l2(lv)

        f = np.concatenate([fusion_cfg.rdf2vec_weight * rv, fusion_cfg.literal_weight * lv], axis=0)
        fused[gu] = _l2(f)
    return fused

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.allclose(a, 0) or np.allclose(b, 0):
        return float("nan")
    return 1.0 - cosine(a, b)

def rank_against_anchor(anchor_uri: str, fused_vecs: Dict[str, np.ndarray]) -> List[Tuple[str, float]]:
    a = fused_vecs[anchor_uri]
    out = []
    for gu, v in fused_vecs.items():
        if gu == anchor_uri:
            continue
        out.append((gu, float(cosine_sim(a, v))))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# -----------------------------
# Top-level API for your backend: embed+rank
# -----------------------------
def embed_and_rank_graphs_hybrid(
    fuseki_endpoint: str,
    graphs: List[str],
    init_graph_uri: str,
    *,
    method: str = "A",  # "A" or "B"
    auth=None,
    rdf_cfg: Optional[RDF2VecConfig] = None,
    fusion_cfg: Optional[FusionConfig] = None,
) -> Dict[str, Any]:
    if not graphs:
        return {"ranking": [], "count": 0}
    if not init_graph_uri:
        raise ValueError("init_graph_uri must be provided.")

    graphs_to_use = list(dict.fromkeys([*graphs, init_graph_uri]))

    ttl_by_uri = {gu: download_graph_ttl(fuseki_endpoint, gu, auth=auth) for gu in graphs_to_use}

    rdf_cfg = rdf_cfg or RDF2VecConfig()
    fusion_cfg = fusion_cfg or FusionConfig()

    if method.upper() == "A":
        literal_encoder = LiteralValueEncoderA(LiteralAConfig())
    elif method.upper() == "B":
        literal_encoder = LiteralValueEncoderB(LiteralBConfig())
    else:
        raise ValueError("method must be 'A' or 'B'.")

    fused = build_fused_vectors(ttl_by_uri, rdf_cfg, literal_encoder, fusion_cfg)
    if init_graph_uri not in fused:
        raise ValueError(f"init_graph_uri '{init_graph_uri}' has no fused vector.")

    ranking = [{"graph_uri": gu, "similarity": sim} for gu, sim in rank_against_anchor(init_graph_uri, fused)]
    return {"ranking": ranking, "count": len(ranking)}
