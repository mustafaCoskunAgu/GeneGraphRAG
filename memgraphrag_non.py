"""
MemGraphRAG with retrieval implemented as ranking on a Network of Networks (NoN).

Indexing is unchanged (OpenIE -> three-layer memory -> schema -> ontology filter ->
conflict detection / resolution -> passage/entity/type graph).

Retrieval replaces Personalized PageRank with CrossRank (Ni, Tong, Fan, Zhang, KDD 2014),
following CR_Precomputation.m, CR.m and J_CR.m:

    NoN R = <G, A, theta>
      G      main network   = type nodes, weighted by type_relation edges
      A_i    domain network = entities of type i + the passages that mention them
      theta  entity_to_type edges (many-to-many: an entity can live in several domains)

    Anorm = blkdiag(D^-1/2 A_i D^-1/2)
    Ynorm = Dy^-1/2 (Y + Dt) Dy^-1/2         (cross-domain mapping, see NoNCrossRank)
    r    <- (gamma*Anorm + kappa*Ynorm) r + eta*e
    gamma = c/(1+2a), kappa = 2a/(1+2a), eta = (1-c)/(1+2a)

Config fields (optional, read with getattr):
    non_alpha     (default 1.0)   cross-domain consistency weight
    non_c         (default 0.5)   smoothness vs. query preference, must be in [0, 1)
    non_max_iter  (default 100)
    non_epsilon   (default 1e-10)
"""
from __future__ import annotations

import ast
import json
import os
import logging
import time
import threading
import re
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from string import Template
from typing import Union, Optional, List, Set, Dict, Any, Tuple

import numpy as np
import scipy.sparse as sp
import igraph as ig
from tqdm import tqdm

from .Memory import ThreeLayerMemory

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import normalize_triple_entry
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig
from .utils.llm_utils import fix_broken_generated_json

from .prompts.prompt import PROMPT as MEMORY_PROMPTS

logger = logging.getLogger(__name__)

PASSAGE, ENTITY, TYPE = 0, 1, 2


# =====================================================================
#  CrossRank on a Network of Networks
# =====================================================================
class NoNCrossRank:
    """
    CrossRank power method with Anorm / Ynorm built like CR_Precomputation.m.

    W        : (n x n) sparse symmetric adjacency of the whole graph
    domains  : list of int arrays; domains[i] = global node ids of domain network A_i
               (domains may overlap: many-to-many mapping theta)
    G        : (g x g) sparse symmetric main network over the domains
    alpha, c : CrossRank parameters (c in [0, 1))
    """

    def __init__(self, W, domains, G, alpha: float = 1.0, c: float = 0.5):
        assert 0.0 <= c < 1.0, "c must be in [0, 1) for convergence"
        self.n, self.alpha, self.c = W.shape[0], float(alpha), float(c)

        # drop empty domains, keep G aligned with the remaining ones
        keep = [i for i, d in enumerate(domains) if len(d) > 0]
        domains = [np.asarray(domains[i], dtype=np.int64) for i in keep]
        G = sp.csr_matrix(G)[keep][:, keep].tocsr()

        sizes = [len(d) for d in domains]
        self.node_of = np.concatenate(domains) if domains else np.zeros(0, dtype=np.int64)
        self.dom_of = np.repeat(np.arange(len(domains)), sizes)
        N = len(self.node_of)
        self.num_domains = len(domains)

        # ---------- Anorm = blkdiag( D^-1/2 A_i D^-1/2 ) ----------
        W = W.tocsr()
        self.Anorm = sp.block_diag(
            [self._sym_norm(W[d][:, d].tocsr()) for d in domains], format="csr")

        # ---------- Y: G(i,j) between the copies of every node shared by domains i and j ----------
        P = sp.csr_matrix((np.ones(N), (np.arange(N), self.node_of)), shape=(N, self.n))
        S = (P @ P.T).tocoo()                         # S[a,b] = 1 if slots a, b hold the same node
        off = S.row != S.col
        ra, cb = S.row[off], S.col[off]
        if len(ra) > 0:
            w = np.asarray(G[self.dom_of[ra], self.dom_of[cb]]).ravel()
        else:
            w = np.zeros(0)
        Y = sp.csr_matrix((w, (ra, cb)), shape=(N, N))

        # ---------- Dt = Dy - diag(rowsum(Y)): every row of Y now sums to dG(i) ----------
        dG = np.asarray(G.sum(axis=1)).ravel()        # main-network degree of each domain
        d_slot = dG[self.dom_of]                      # Dy = dG(i) * I
        Y = Y + sp.diags(d_slot - np.asarray(Y.sum(axis=1)).ravel())

        # ---------- Ynorm = Dy^-1/2 Y Dy^-1/2 ----------
        inv = np.zeros(N)
        nz = d_slot > 0
        inv[nz] = d_slot[nz] ** -0.5
        Dn = sp.diags(inv)
        # domains with dG = 0 (no neighbour in G; MATLAB would give Inf/NaN) keep their own score
        self.Ynorm = (Dn @ Y @ Dn + sp.diags((~nz).astype(float))).tocsr()

        # ---------- constants from CR.m ----------
        self.gamma = self.c / (1 + 2 * self.alpha)
        self.kappa = 2 * self.alpha / (1 + 2 * self.alpha)
        self.eta = (1 - self.c) / (1 + 2 * self.alpha)
        self.M = (self.gamma * self.Anorm + self.kappa * self.Ynorm).tocsr()   # built once

    # ------------------------------------------------------------------
    @classmethod
    def from_layered_graph(cls, W, layer_of, edge_list, edge_weights,
                           alpha: float = 1.0, c: float = 0.5) -> "NoNCrossRank":
        """
        Build the NoN from a passage / entity / type graph.

        W            : (n x n) sparse symmetric adjacency
        layer_of     : (n,) array with PASSAGE / ENTITY / TYPE (or -1 for unknown)
        edge_list    : list of (u, v) vertex-index pairs
        edge_weights : weight of every edge
        """
        n = W.shape[0]
        type_idxs = np.flatnonzero(layer_of == TYPE)
        type2dom = {int(t): i for i, t in enumerate(type_idxs)}

        ent2passages: Dict[int, Set[int]] = defaultdict(set)
        members: Dict[int, Set[int]] = defaultdict(set)
        gr, gc, gv = [], [], []
        for (a, b), w in zip(edge_list, edge_weights):
            la, lb = layer_of[a], layer_of[b]
            if la == TYPE and lb == TYPE:
                if a != b:
                    i, j = type2dom[a], type2dom[b]          # G(i, j) from type_relation edges
                    gr += [i, j]; gc += [j, i]; gv += [w, w]
                continue
            for x, y in ((a, b), (b, a)):
                if layer_of[x] == PASSAGE and layer_of[y] == ENTITY:
                    ent2passages[y].add(x)
                elif layer_of[x] == ENTITY and layer_of[y] == TYPE:
                    members[type2dom[y]].add(x)              # theta: entity belongs to type domain

        domains, covered = [], set()
        for d in range(len(type_idxs)):
            idxs = set()
            for ent in members[d]:
                idxs.add(ent)
                idxs.update(ent2passages.get(ent, ()))
            covered |= idxs
            domains.append(np.array(sorted(idxs), dtype=np.int64))

        # nodes without any type -> one extra domain so nothing is unreachable
        leftovers = [i for i in range(n) if layer_of[i] != TYPE and i not in covered]
        domains.append(np.array(leftovers, dtype=np.int64))

        g = len(domains)
        G = sp.coo_matrix((np.array(gv, dtype=float),
                           (np.array(gr, dtype=np.int64), np.array(gc, dtype=np.int64))),
                          shape=(g, g)).tocsr()
        return cls(W, domains, G, alpha=alpha, c=c)

    @staticmethod
    def _sym_norm(S):
        deg = np.asarray(S.sum(axis=1)).ravel()
        inv = np.zeros_like(deg, dtype=float)
        nz = deg > 0                                  # isolated nodes: MATLAB gives Inf*0 = NaN
        inv[nz] = deg[nz] ** -0.5
        D = sp.diags(inv)
        return (D @ S @ D).tocsr()

    def objective(self, r: np.ndarray, e: np.ndarray) -> float:
        """J_CR.m:  c r'(I-A)r + (1-c)||r-e||^2 + 2a r'(I-Y)r"""
        rr = r @ r
        return float(self.c * (rr - r @ (self.Anorm @ r))
                     + (1 - self.c) * ((r - e) @ (r - e))
                     + 2 * self.alpha * (rr - r @ (self.Ynorm @ r)))

    def run(self, seed: np.ndarray, max_iter: int = 100, epsilon: float = 1e-10) -> np.ndarray:
        """seed: (n,) query over global nodes. Returns (n,) score per global node."""
        s = np.where(np.isnan(seed) | (seed < 0), 0, seed).astype(float)
        s = s / s.sum() if s.sum() > 0 else np.full(self.n, 1.0 / self.n)

        e = s[self.node_of]                           # query copied into every domain of a node
        r = e.copy()                                  # r = e (initialization in CR.m)
        J = self.objective(r, e)
        delta, it = np.inf, 1
        while delta > epsilon and it <= max_iter:     # same stopping rule as CR.m
            r = self.M @ r + self.eta * e             # r = M*r + eta*e
            J_new = self.objective(r, e)
            delta, J = J - J_new, J_new
            it += 1

        scores = np.bincount(self.node_of, weights=r, minlength=self.n)   # merge the domains
        total = scores.sum()
        return scores / total if total > 0 else scores

    def run_exact(self, seed: np.ndarray) -> np.ndarray:
        """Direct solve of (I - M) r = eta e. Use as a correctness check for `run`."""
        from scipy.sparse.linalg import spsolve
        s = np.where(np.isnan(seed) | (seed < 0), 0, seed).astype(float)
        s = s / s.sum() if s.sum() > 0 else np.full(self.n, 1.0 / self.n)
        e = s[self.node_of]
        N = len(e)
        r = spsolve((sp.identity(N, format="csc") - self.M).tocsc(), self.eta * e)
        scores = np.bincount(self.node_of, weights=r, minlength=self.n)
        return scores / scores.sum()


# =====================================================================
#  MemGraphRAG
# =====================================================================
class MemGraphRAG_NON:

    def __init__(self, global_config=None, save_dir=None, llm_model_name=None, embedding_model_name=None, llm_base_url=None):
        """
        Initializes the instance and its components (LLM, OpenIE, embedding stores, graph).

        Parameters:
            global_config: BaseConfig instance. A default BaseConfig is created if None.
            save_dir: directory for this instance's files (overrides the config).
            llm_model_name / embedding_model_name / llm_base_url: optional overrides of the config.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # Overwriting configuration if specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"MemGraphRAG_NON init with config:\n  {_print_config}\n")

        # LLM and embedding specific working directories are created under the saving directory
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            # Lazy import: vllm pulls transformers->torchvision; skip when using online OpenIE only.
            from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE

            self.openie = VLLMOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir, f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        # Entity to triples index for fast lookup
        # Key: processed entity string, Value: List of dicts with 'chunk_id', 'triple', 'passage'
        self.entity_to_triples: Dict[str, List[Dict[str, Any]]] = {}

        # openie results
        self.openie_results = None
        # memory
        self.memory = None
        self.memory_path = os.path.join(self.global_config.save_dir, "initial_memory.json")

        # Network-of-Networks ranker, built in prepare_retrieval_objects
        self.non: Optional[NoNCrossRank] = None

    def initialize_graph(self):
        """Load graph.graphml if available (and not forced from scratch), else create an empty graph."""
        self._graphml_xml_file = os.path.join(
            self.working_dir, f"graph.graphml"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graphml_xml_file):
                preloaded_graph = ig.Graph.Read_GraphML(self._graphml_xml_file)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self, docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        self.openie_results = all_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def triple_extraction(self, docs: List[str]):
        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            httpx_logger = logging.getLogger("httpx")
            previous_httpx_level = httpx_logger.level
            try:
                httpx_logger.setLevel(logging.WARNING)
                new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            finally:
                httpx_logger.setLevel(previous_httpx_level)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)
        self.openie_results = all_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

    def build_memory(self):
        # 先从指定路径加载openie结果
        with open(self.openie_results_path, "r", encoding="utf-8") as f:
            openie_results = json.load(f)
        memory = ThreeLayerMemory()
        memory.build_from_raw_openie_results(openie_results)
        memory.save(self.memory_path)
        self.memory = memory

    @staticmethod
    def _memory_json(response: str) -> Dict[str, Any]:
        """Parse JSON returned by a memory-pipeline prompt."""
        cleaned = str(response).strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            candidate = match.group(0) if match else cleaned
            return json.loads(fix_broken_generated_json(candidate))

    @staticmethod
    def _memory_infer_result(result: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(result, (tuple, list)):
            text = str(result[0]) if result else ""
            metadata = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
            return text, metadata
        return str(result), {}

    @staticmethod
    def _normalize_memory_text(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @classmethod
    def _normalize_memory_triple(cls, triple: Any) -> Tuple[str, str, str]:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return "", "", ""
        return tuple(cls._normalize_memory_text(x) for x in triple)

    @staticmethod
    def _memory_to_dict(memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> Dict[str, Any]:
        return memory.to_dict() if isinstance(memory, ThreeLayerMemory) else json.loads(json.dumps(memory))

    @staticmethod
    def _write_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    #  Memory pipeline: schema, ontology filter, conflicts
    # ------------------------------------------------------------------
    def extract_memory_schema(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> ThreeLayerMemory:
        """Attach one ontology/schema triple to every fact using passage context."""
        data = self._memory_to_dict(memory)
        facts = data.get("fact_layer", [])
        passages = data.get("passage_layer", [])
        schemas = data.get("schema_layer", [])
        schema_to_idx = {
            tuple(map(str, row.get("content", []))): int(row["idx"])
            for row in schemas if len(row.get("content", [])) == 3
        }

        def extract_one(fact: Dict[str, Any]) -> Tuple[int, Optional[Tuple[str, str, str]]]:
            triple = fact.get("content", [])
            contexts = []
            for passage_idx in fact.get("passage_indices", [])[:3]:
                if isinstance(passage_idx, int) and 0 <= passage_idx < len(passages):
                    text = str(passages[passage_idx].get("content", "")).strip()
                    if text:
                        contexts.append(text)
            formatted = "\n".join(
                f"- ({t[0]}) [{t[1]}] ({t[2]})" for t in [triple]
                if isinstance(t, list) and len(t) == 3
            )
            prompt = MEMORY_PROMPTS["ontology_extraction"]
            user = prompt["user"].replace("${passage}", "\n\n".join(contexts) or "No passage context provided.")
            user = user.replace("${triples}", formatted)
            result = self.llm_model.infer(
                [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}],
                max_completion_tokens=self.global_config.schema_extraction_max_new_tokens,
                temperature=self.global_config.schema_extraction_temperature,
            )
            response, _ = self._memory_infer_result(result)
            rows = self._memory_json(response).get("ontology_triples", [])
            if not rows:
                return int(fact["idx"]), None
            ontology = rows[0].get("ontology", [])
            if not isinstance(ontology, list) or len(ontology) != 3:
                return int(fact["idx"]), None
            normalized = []
            for position, value in enumerate(ontology):
                value = str(value).strip() or "Unknown"
                if position != 1 and not (value.startswith("<") and value.endswith(">")):
                    value = f"<{value}>"
                normalized.append(value)
            return int(fact["idx"]), tuple(normalized)

        pending = [
            fact for fact in facts
            if self.global_config.schema_extraction_force_reextract
            or int(fact.get("schema_idx", -1)) < 0
        ]
        sample = self.global_config.schema_extraction_sample_facts
        if sample > 0:
            pending = pending[:sample]
        extracted: Dict[int, Tuple[str, str, str]] = {}
        workers = max(1, self.global_config.memory_max_workers)
        if workers == 1:
            for fact in tqdm(pending, desc="Extracting schema"):
                fact_idx, ontology = extract_one(fact)
                if ontology is not None:
                    extracted[fact_idx] = ontology
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(extract_one, fact) for fact in pending]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting schema"):
                    fact_idx, ontology = future.result()
                    if ontology is not None:
                        extracted[fact_idx] = ontology

        for fact in facts:
            ontology = extracted.get(int(fact.get("idx", -1)))
            if ontology is None:
                continue
            if ontology not in schema_to_idx:
                schema_idx = len(schemas)
                schema_to_idx[ontology] = schema_idx
                schemas.append({"idx": schema_idx, "content": list(ontology), "frequency": 0,
                                "embedding": None, "fact_indices": []})
            fact["schema_idx"] = schema_to_idx[ontology]

        by_schema: Dict[int, List[int]] = defaultdict(list)
        fact_frequency = {int(fact["idx"]): int(fact.get("frequency", 0) or 0) for fact in facts}
        for fact in facts:
            schema_idx = int(fact.get("schema_idx", -1))
            if schema_idx >= 0:
                by_schema[schema_idx].append(int(fact["idx"]))
        for schema in schemas:
            fact_indices = sorted(set(by_schema.get(int(schema["idx"]), [])))
            schema["fact_indices"] = fact_indices
            schema["frequency"] = sum(fact_frequency.get(fact_idx, 0) for fact_idx in fact_indices)
        data["schema_layer"] = schemas
        data["stats"] = {"num_schemas": len(schemas), "num_facts": len(facts), "num_passages": len(passages)}
        return ThreeLayerMemory.from_dict(data)

    def filter_memory_ontology(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> ThreeLayerMemory:
        """Remove low-frequency schemas and rebuild all inter-layer indices."""
        source = ThreeLayerMemory.from_dict(self._memory_to_dict(memory))
        mode = self.global_config.ontology_filter_mode
        if mode == "absolute":
            removed = {s.idx for s in source.schema_layer if s.frequency < self.global_config.ontology_filter_min_frequency}
        elif mode == "percentile":
            percent = min(100.0, max(0.0, self.global_config.ontology_filter_low_percent))
            count = int(len(source.schema_layer) * percent / 100.0)
            removed = {s.idx for s in sorted(source.schema_layer, key=lambda s: (s.frequency, s.idx))[:count]}
        else:
            raise ValueError(f"Unsupported ontology_filter_mode: {mode}")

        data = source.to_dict()
        kept_schemas = [s for s in data["schema_layer"] if int(s["idx"]) not in removed]
        schema_map = {int(row["idx"]): idx for idx, row in enumerate(kept_schemas)}
        kept_facts = [f for f in data["fact_layer"] if int(f.get("schema_idx", -1)) in schema_map]
        fact_map = {int(row["idx"]): idx for idx, row in enumerate(kept_facts)}
        for idx, row in enumerate(kept_facts):
            row["idx"] = idx
            row["schema_idx"] = schema_map[int(row["schema_idx"])]
            row["passage_indices"] = list(dict.fromkeys(row.get("passage_indices", [])))
            row["frequency"] = len(row["passage_indices"])
        for idx, row in enumerate(kept_schemas):
            row["idx"] = idx
            row["fact_indices"] = [fact_map[f] for f in row.get("fact_indices", []) if f in fact_map]
            row["frequency"] = len(row["fact_indices"])
        for row in data["passage_layer"]:
            row["fact_indices"] = [fact_map[f] for f in row.get("fact_indices", []) if f in fact_map]
        result = {"schema_layer": kept_schemas, "fact_layer": kept_facts, "passage_layer": data["passage_layer"],
                  "stats": {"num_schemas": len(kept_schemas), "num_facts": len(kept_facts),
                            "num_passages": len(data["passage_layer"])}}
        return ThreeLayerMemory.from_dict(result)

    def _conflict_maps(self, memory: ThreeLayerMemory) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, int]]:
        facts, evidence, order = {}, {}, {}
        limit = self.global_config.conflict_passage_evidence_per_fact
        for position, fact in enumerate(memory.fact_layer):
            facts[fact.idx] = list(fact.content)
            order[fact.idx] = position
            evidence[fact.idx] = [str(p.content)[:1200] for p in memory.get_passages_by_fact(fact.idx)[:limit]]
        return facts, evidence, order

    def _conflict_candidates(self, facts: Dict[int, List[str]]) -> Dict[int, List[int]]:
        by_subject_relation: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        duplicates: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        for fact_idx, triple in facts.items():
            normalized = self._normalize_memory_triple(triple)
            by_subject_relation[normalized[:2]].append(fact_idx)
            duplicates[normalized].append(fact_idx)
        duplicate_peers: Dict[int, Set[int]] = defaultdict(set)
        for ids in duplicates.values():
            for fact_idx in ids:
                duplicate_peers[fact_idx].update(set(ids) - {fact_idx})
        candidates: Dict[int, Set[int]] = {fact_idx: set() for fact_idx in facts}
        for ids in by_subject_relation.values():
            for fact_idx in ids:
                own = self._normalize_memory_triple(facts[fact_idx])
                candidates[fact_idx].update(
                    other for other in ids
                    if other != fact_idx and self._normalize_memory_triple(facts[other])[2] != own[2]
                )
        if self.global_config.conflict_enable_reverse_relation_check:
            allowed = {self._normalize_memory_text(x) for x in self.global_config.conflict_reverse_relations}
            for fact_idx, triple in facts.items():
                head, relation, tail = self._normalize_memory_triple(triple)
                if relation not in allowed:
                    continue
                for other, other_triple in facts.items():
                    other_head, other_relation, other_tail = self._normalize_memory_triple(other_triple)
                    if other != fact_idx and relation == other_relation and head == other_tail and tail == other_head:
                        candidates[fact_idx].add(other)
        limit = self.global_config.conflict_max_related_per_target
        return {key: [x for x in sorted(value) if x not in duplicate_peers[key]][:limit]
                for key, value in candidates.items()}

    def detect_memory_conflicts(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> Dict[str, Any]:
        """Detect hard conflicts among candidate facts."""
        memory_obj = ThreeLayerMemory.from_dict(self._memory_to_dict(memory))
        facts, evidence, order = self._conflict_maps(memory_obj)
        candidates = self._conflict_candidates(facts)

        def detect_one(target_id: int, related_ids: List[int]) -> Dict[str, Any]:
            related_lines = []
            if self.global_config.conflict_include_passage_evidence:
                related_lines.extend(f"Target evidence {i}: {text}" for i, text in enumerate(evidence[target_id], 1))
            for rank, related_id in enumerate(related_ids, 1):
                related_lines.append(f"- Triple {rank} | ID: {related_id} | triple: {facts[related_id]}")
                if self.global_config.conflict_include_passage_evidence:
                    related_lines.extend(f"  Evidence {i}: {text}" for i, text in enumerate(evidence[related_id], 1))
            prompt = MEMORY_PROMPTS["conflict_detection"]
            user = Template(prompt["user"]).substitute(
                target_triple=str(facts[target_id]), related_triples="\n".join(related_lines))
            raw = self.llm_model.infer(
                [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}],
                max_completion_tokens=self.global_config.conflict_detection_max_new_tokens,
                temperature=0.0,
            )
            response, metadata = self._memory_infer_result(raw)
            parsed = self._memory_json(response)
            hard = []
            matched_ids: Set[int] = set()
            for value in parsed.get("conflicting_triple_ids", []):
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value in related_ids:
                    matched_ids.add(value)
            for entry in parsed.get("conflicts", []):
                if not isinstance(entry, dict):
                    continue
                conflict_type = str(entry.get("conflict_type", "uncertain")).lower()
                confidence = float(entry.get("confidence", 0.0) or 0.0)
                if (not entry.get("is_hard_conflict") and not entry.get("needs_resolution")) \
                        or confidence < self.global_config.conflict_min_confidence \
                        or conflict_type in {"duplicate", "none", "uncertain"}:
                    continue
                other_id = None
                for field in ("triple2", "other_triple"):
                    normalized = self._normalize_memory_triple(entry.get(field))
                    for candidate_id in related_ids:
                        if normalized == self._normalize_memory_triple(facts[candidate_id]):
                            other_id = candidate_id
                            break
                if other_id is not None:
                    entry["_matched_other_id"] = other_id
                    matched_ids.add(other_id)
                hard.append(entry)
            return {"target_id": target_id, "related_ids": related_ids, "hard_conflicts": hard,
                    "conflicting_related_ids": sorted(matched_ids),
                    "token_usage": {"prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                                    "completion_tokens": int(metadata.get("completion_tokens", 0) or 0)}}

        tasks = []
        for fact_idx, related in candidates.items():
            if self.global_config.conflict_streaming_only_previous:
                related = [other for other in related if order[other] < order[fact_idx]]
            if related:
                tasks.append((fact_idx, related))
        results = []
        workers = max(1, self.global_config.memory_max_workers)
        if workers == 1:
            results = [detect_one(*task) for task in tqdm(tasks, desc="Detecting conflicts")]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(detect_one, *task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Detecting conflicts"):
                    results.append(future.result())

        items, seen, conflicting = [], set(), set()
        for result in results:
            target_id = int(result["target_id"])
            detailed = {int(x.get("_matched_other_id")): x for x in result["hard_conflicts"]
                        if x.get("_matched_other_id") is not None}
            for other_id in result["conflicting_related_ids"]:
                pair = tuple(sorted((target_id, int(other_id))))
                if pair in seen:
                    continue
                seen.add(pair)
                entry = detailed.get(int(other_id), {})
                items.append({"target_fact_id": target_id, "other_fact_id": int(other_id),
                              "target_triple": facts[target_id], "other_triple": facts[int(other_id)],
                              "conflict_type": entry.get("conflict_type", "unspecified"),
                              "confidence": float(entry.get("confidence", 0.6) or 0.6),
                              "reason": entry.get("conflict_reason", "Returned by model."),
                              "needs_resolution": True,
                              "evidence_passages": {"target": evidence[target_id], "other": evidence[int(other_id)]}})
                conflicting.update(pair)
        prompt_tokens = sum(r["token_usage"]["prompt_tokens"] for r in results)
        completion_tokens = sum(r["token_usage"]["completion_tokens"] for r in results)
        return {"meta": {"created_at": datetime.now().isoformat(timespec="seconds"),
                         "llm_name": self.global_config.llm_name,
                         "max_workers": workers},
                "summary": {"num_facts": len(facts), "num_detection_tasks": len(tasks),
                            "num_candidate_edges": sum(map(len, candidates.values())),
                            "num_conflict_items": len(items), "num_conflicting_facts": len(conflicting),
                            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens},
                "conflict_items": items}

    def _apply_memory_actions(self, memory: ThreeLayerMemory, actions: Dict[int, Dict[str, Any]]) -> ThreeLayerMemory:
        """Apply conflict actions and rebuild every inter-layer index."""
        source = memory.to_dict()
        facts = []
        old_to_new: Dict[int, int] = {}
        for fact in source["fact_layer"]:
            old_idx = int(fact["idx"])
            action = actions.get(old_idx, {})
            if action.get("action") == "discarded":
                continue
            new_fact = dict(fact)
            resolved = action.get("resolved_triple")
            if action.get("action") == "modified" and isinstance(resolved, (list, tuple)) and len(resolved) == 3:
                new_fact["content"] = list(map(str, resolved))
                if self.global_config.conflict_resolution_set_schema_negative_on_modified:
                    new_fact["schema_idx"] = -1
            new_idx = len(facts)
            old_to_new[old_idx] = new_idx
            new_fact["idx"] = new_idx
            new_fact["passage_indices"] = sorted({int(x) for x in new_fact.get("passage_indices", [])
                                                  if 0 <= int(x) < len(source["passage_layer"])})
            new_fact["frequency"] = len(new_fact["passage_indices"])
            facts.append(new_fact)
        schemas = [dict(row) for row in source["schema_layer"]]
        by_schema: Dict[int, List[int]] = defaultdict(list)
        for fact in facts:
            schema_idx = int(fact.get("schema_idx", -1))
            if 0 <= schema_idx < len(schemas):
                by_schema[schema_idx].append(int(fact["idx"]))
        for schema in schemas:
            schema_idx = int(schema["idx"])
            schema["fact_indices"] = sorted(by_schema.get(schema_idx, []))
            schema["frequency"] = len(schema["fact_indices"])
        passages = []
        for passage in source["passage_layer"]:
            row = dict(passage)
            row["fact_indices"] = sorted({old_to_new[x] for x in passage.get("fact_indices", []) if x in old_to_new})
            passages.append(row)
        data = {"schema_layer": schemas, "fact_layer": facts, "passage_layer": passages,
                "stats": {"num_schemas": len(schemas), "num_facts": len(facts), "num_passages": len(passages)}}
        return ThreeLayerMemory.from_dict(data)

    def resolve_memory_conflicts(
        self, memory: Union[ThreeLayerMemory, Dict[str, Any]], conflict_data: Dict[str, Any]
    ) -> Tuple[ThreeLayerMemory, Dict[str, Any]]:
        """Resolve connected conflict components and return rebuilt memory plus audit data."""
        memory_obj = ThreeLayerMemory.from_dict(self._memory_to_dict(memory))
        facts, evidence, _ = self._conflict_maps(memory_obj)
        edges = []
        for item in conflict_data.get("conflict_items", []):
            try:
                left, right = int(item["target_fact_id"]), int(item["other_fact_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if item.get("needs_resolution") and left in facts and right in facts and left != right:
                edges.append((left, right, item))
        adjacency: Dict[int, Set[int]] = defaultdict(set)
        for left, right, _ in edges:
            adjacency[left].add(right)
            adjacency[right].add(left)
        components, visited = [], set()
        for start in adjacency:
            if start in visited:
                continue
            queue, component = deque([start]), []
            visited.add(start)
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(sorted(component))

        def resolve_component(component: List[int]) -> Dict[str, Any]:
            if len(component) > self.global_config.conflict_resolution_max_component_size:
                return {"component_fact_ids": component, "resolved_triples": [],
                        "unresolved_conflicts": [{"triple_ids": component, "reason": "component too large"}],
                        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}
            rows = []
            component_set = set(component)
            for left, right, item in edges:
                if left in component_set and right in component_set:
                    rows.append({"triple_id_1": left, "triple_1": facts[left],
                                 "triple_id_2": right, "triple_2": facts[right],
                                 "detected_conflict_type": item.get("conflict_type", "unspecified"),
                                 "detected_reason": item.get("reason", ""),
                                 "evidence_1": evidence[left], "evidence_2": evidence[right]})
            prompt = MEMORY_PROMPTS["conflict_resolution"]
            payloads = [(prompt["system"], rows)]
            if self.global_config.conflict_resolution_enable_content_filter_fallback:
                minimal = [{key: value for key, value in row.items()
                            if not key.startswith("evidence") and key != "detected_reason"} for row in rows]
                payloads.append(("Resolve knowledge graph conflicts conservatively. Output JSON only.", minimal))
            last_error = None
            for attempt, (system, payload) in enumerate(payloads, 1):
                try:
                    user = Template(prompt["user"]).substitute(
                        conflicting_triples_with_sources=json.dumps(
                            {"component_fact_ids": component, "conflicts": payload}, ensure_ascii=False, indent=2))
                    raw = self.llm_model.infer(
                        [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_completion_tokens=self.global_config.conflict_resolution_max_new_tokens,
                        temperature=0.0,
                    )
                    response, metadata = self._memory_infer_result(raw)
                    parsed = self._memory_json(response)
                    parsed["component_fact_ids"] = component
                    parsed["fallback_attempt"] = attempt
                    parsed["token_usage"] = {"prompt_tokens": int(metadata.get("prompt_tokens", 0) or 0),
                                             "completion_tokens": int(metadata.get("completion_tokens", 0) or 0)}
                    return parsed
                except Exception as exc:
                    last_error = exc
                    message = str(exc).lower()
                    if "content_filter" not in message and "content management policy" not in message:
                        raise
            return {"component_fact_ids": component, "resolved_triples": [],
                    "unresolved_conflicts": [{"triple_ids": component, "reason": str(last_error)}],
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}

        component_results = [resolve_component(component) for component in tqdm(components, desc="Resolving conflicts")]
        rank = {"kept": 1, "modified": 2, "discarded": 3}
        actions: Dict[int, Dict[str, Any]] = {}
        by_normalized = defaultdict(list)
        for fact_idx, triple in facts.items():
            by_normalized[self._normalize_memory_triple(triple)].append(fact_idx)
        for result in component_results:
            for row in result.get("resolved_triples", []):
                action = str(row.get("resolution", "")).lower()
                if action not in rank:
                    continue
                fact_idx = None
                try:
                    candidate = int(row.get("triple_id"))
                    fact_idx = candidate if candidate in facts else None
                except (TypeError, ValueError):
                    pass
                if fact_idx is None:
                    matches = by_normalized[self._normalize_memory_triple(row.get("original_triple"))]
                    fact_idx = matches[0] if len(matches) == 1 else None
                if fact_idx is None:
                    continue
                resolved = row.get("resolved_triple")
                info = {"action": action,
                        "resolved_triple": list(map(str, resolved)) if isinstance(resolved, (list, tuple)) and len(resolved) == 3 else None,
                        "reason": str(row.get("reason", "")), "conflict_type": row.get("conflict_type", "unspecified")}
                if fact_idx not in actions or rank[action] > rank[actions[fact_idx]["action"]]:
                    actions[fact_idx] = info
        updated = self._apply_memory_actions(memory_obj, actions)
        counts = {name: sum(1 for item in actions.values() if item["action"] == name) for name in rank}
        prompt_tokens = sum(x.get("token_usage", {}).get("prompt_tokens", 0) for x in component_results)
        completion_tokens = sum(x.get("token_usage", {}).get("completion_tokens", 0) for x in component_results)
        audit = {"meta": {"created_at": datetime.now().isoformat(timespec="seconds"),
                          "llm_name": self.global_config.llm_name,
                          "max_component_size": self.global_config.conflict_resolution_max_component_size},
                 "summary": {"num_conflict_items_input": len(conflict_data.get("conflict_items", [])),
                             "num_resolvable_edges": len(edges), "num_components": len(components),
                             "actions_applied": counts, "facts_before": len(facts),
                             "facts_after": len(updated.fact_layer), "prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens},
                 "component_resolutions": component_results,
                 "applied_actions": [{"fact_id": fact_idx, **info} for fact_idx, info in sorted(actions.items())]}
        return updated, audit

    # ------------------------------------------------------------------
    #  Final passage / entity / type graph
    # ------------------------------------------------------------------
    def build_memory_graph(self, memory: Union[ThreeLayerMemory, Dict[str, Any]]) -> Dict[str, Any]:
        """Build the final type/entity/passage graph from conflict-resolved memory."""
        data = self._memory_to_dict(memory)
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        def add_node(node_id: str, **attrs: Any) -> None:
            nodes.setdefault(node_id, {"id": node_id, **attrs})

        def add_edge(source: str, target: str, edge_type: str, weight: float = 1.0,
                     accumulate: bool = False, **attrs: Any) -> None:
            if source == target:
                return
            key = (source, target, edge_type)
            if key not in edges:
                edges[key] = {"source": source, "target": target, "type": edge_type,
                              "weight": float(weight), **attrs}
            elif accumulate:
                edges[key]["weight"] += float(weight)

        schemas = {int(row["idx"]): row for row in data.get("schema_layer", [])}
        passages = {int(row["idx"]): row for row in data.get("passage_layer", [])}
        final_entities: Set[str] = set()
        final_facts = []
        entity_types: Dict[str, Set[str]] = defaultdict(set)

        # Keep every indexed passage node, including chunks without a surviving fact.
        for chunk_id, row in self.chunk_embedding_store.get_text_for_all_rows().items():
            content = str(row.get("content", ""))
            add_node(chunk_id, layer="passage", content=content, label=content[:120])

        for fact in data.get("fact_layer", []):
            triple = fact.get("content", [])
            if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                continue
            head, _, tail = map(str, triple)
            final_entities.update((head, tail))
            final_facts.append(str((head, str(triple[1]), tail)))
            head_id = compute_mdhash_id(head, prefix="entity-")
            tail_id = compute_mdhash_id(tail, prefix="entity-")
            add_node(head_id, layer="entity", content=head, label=head)
            add_node(tail_id, layer="entity", content=tail, label=tail)
            # Preserve HippoRAG entity-to-entity weighting: one count per surviving fact.
            add_edge(head_id, tail_id, "entity_relation", accumulate=True)
            add_edge(tail_id, head_id, "entity_relation", accumulate=True)

            for passage_idx in fact.get("passage_indices", []):
                passage = passages.get(int(passage_idx))
                if passage is None:
                    continue
                passage_id = str(passage.get("chunk_id", ""))
                if passage_id in nodes:
                    add_edge(passage_id, head_id, "passage_entity")
                    add_edge(passage_id, tail_id, "passage_entity")

            schema = schemas.get(int(fact.get("schema_idx", -1)))
            if schema is not None and len(schema.get("content", [])) == 3:
                head_type, _, tail_type = map(str, schema["content"])
                entity_types[head_type].add(head)
                entity_types[tail_type].add(tail)

        self.entity_embedding_store.insert_strings(sorted(final_entities))
        self.fact_embedding_store.insert_strings(final_facts)

        # Type nodes and entity-type edges (theta). Each type distributes total weight 1 over its entities.
        for type_name, entities in entity_types.items():
            type_id = compute_mdhash_id(type_name, prefix="type-")
            add_node(type_id, layer="type", content=type_name, label=type_name,
                     entity_count=len(entities))
            weight = 1.0 / len(entities) if entities else 0.0
            for entity in entities:
                entity_id = compute_mdhash_id(entity, prefix="entity-")
                add_edge(entity_id, type_id, "entity_to_type", weight=weight)
                if self.global_config.is_directed_graph:
                    add_edge(type_id, entity_id, "type_to_entity", weight=weight)

        # Type-to-type weight (main network G) is the sum of frequencies of schemas joining the same type pair.
        for schema in data.get("schema_layer", []):
            content = schema.get("content", [])
            frequency = float(schema.get("frequency", 0) or 0)
            if len(content) != 3 or frequency <= 0:
                continue
            head_type, _, tail_type = map(str, content)
            head_type_id = compute_mdhash_id(head_type, prefix="type-")
            tail_type_id = compute_mdhash_id(tail_type, prefix="type-")
            if head_type_id in nodes and tail_type_id in nodes:
                add_edge(head_type_id, tail_type_id, "type_relation", frequency, accumulate=True)
                if self.global_config.is_directed_graph:
                    add_edge(tail_type_id, head_type_id, "type_relation", frequency, accumulate=True)

        # Add synonymy edges only now, using entities that survived conflict resolution.
        entity_ids = [compute_mdhash_id(entity, prefix="entity-") for entity in sorted(final_entities)]
        if len(entity_ids) > 1:
            embeddings = self.entity_embedding_store.get_embeddings(entity_ids)
            neighbors = retrieve_knn(
                query_ids=entity_ids, key_ids=entity_ids,
                query_vecs=embeddings, key_vecs=embeddings,
                k=min(len(entity_ids), self.global_config.synonymy_edge_topk + 1),
                query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                key_batch_size=self.global_config.synonymy_edge_key_batch_size,
            )
            for source_id, (neighbor_ids, scores) in neighbors.items():
                entity = nodes[source_id]["content"]
                if len(re.sub("[^A-Za-z0-9]", "", entity)) <= 2:
                    continue
                for target_id, score in zip(neighbor_ids, scores):
                    if source_id == target_id or float(score) < self.global_config.synonymy_edge_sim_threshold:
                        continue
                    add_edge(source_id, target_id, "entity_similarity",
                             weight=float(score), similarity=float(score))

        stats = {
            "num_type_nodes": sum(n.get("layer") == "type" for n in nodes.values()),
            "num_passage_nodes": sum(n.get("layer") == "passage" for n in nodes.values()),
            "num_entity_nodes": sum(n.get("layer") == "entity" for n in nodes.values()),
            "num_edges": len(edges),
            "num_entity_relation_edges": sum(e["type"] == "entity_relation" for e in edges.values()),
            "num_passage_entity_edges": sum(e["type"] == "passage_entity" for e in edges.values()),
            "num_entity_type_edges": sum(e["type"] in {"entity_to_type", "type_to_entity"} for e in edges.values()),
            "num_type_relation_edges": sum(e["type"] == "type_relation" for e in edges.values()),
            "num_entity_similarity_edges": sum(e["type"] == "entity_similarity" for e in edges.values()),
        }
        return {"nodes": list(nodes.values()), "edges": list(edges.values()), "stats": stats}

    def install_memory_graph(self, graph: Dict[str, Any]) -> None:
        """Install the final three-node-type graph as the graph used by retrieval."""
        result = ig.Graph(directed=self.global_config.is_directed_graph)
        node_ids = [node["id"] for node in graph["nodes"]]
        result.add_vertices(len(node_ids))
        result.vs["name"] = node_ids
        result.vs["layer"] = [node.get("layer", "") for node in graph["nodes"]]
        result.vs["content"] = [str(node.get("content", "")) for node in graph["nodes"]]
        valid = set(node_ids)
        graph_edges = [edge for edge in graph["edges"]
                       if edge["source"] in valid and edge["target"] in valid]
        result.add_edges([(edge["source"], edge["target"]) for edge in graph_edges])
        result.es["weight"] = [float(edge.get("weight", 1.0)) for edge in graph_edges]
        result.es["type"] = [edge.get("type", "") for edge in graph_edges]
        self.graph = result
        self.node_to_node_stats = {
            (edge["source"], edge["target"]): float(edge.get("weight", 1.0))
            for edge in graph_edges
        }
        entity_passages: Dict[str, Set[str]] = defaultdict(set)
        for edge in graph_edges:
            if edge.get("type") != "passage_entity":
                continue
            entity_passages[edge["target"]].add(edge["source"])
        self.ent_node_to_num_chunk = {
            node["id"]: len(entity_passages.get(node["id"], set()))
            for node in graph["nodes"] if node.get("layer") == "entity"
        }
        self.save_igraph()
        self.ready_to_retrieve = False          # force NoN rebuild on next retrieval

    def save_memory_graph(self, graph: Dict[str, Any]) -> Dict[str, Optional[str]]:
        output_dir = os.path.join(self.global_config.save_dir, "graph_from_memory")
        stem = self.global_config.memory_graph_output_name
        json_path = os.path.join(output_dir, f"{stem}.json")
        graphml_path = os.path.join(output_dir, f"{stem}.graphml")
        self._write_json(json_path, graph)
        os.makedirs(output_dir, exist_ok=True)
        self.graph.write_graphml(graphml_path)
        return {"json": json_path, "graphml": graphml_path}

    def index_with_memory(self, docs: List[str]) -> Dict[str, Any]:
        """Run OpenIE, memory construction, conflict resolution, and final graph construction."""
        if not docs:
            raise ValueError("index_with_memory requires at least one document")
        missing_prompts = {"ontology_extraction", "conflict_detection", "conflict_resolution"} - set(MEMORY_PROMPTS)
        if missing_prompts:
            raise RuntimeError(f"Missing memory prompts: {sorted(missing_prompts)}")
        detailed = self.global_config.memory_artifact_mode == "detailed"
        if self.global_config.memory_artifact_mode not in {"default", "detailed"}:
            raise ValueError("memory_artifact_mode must be 'default' or 'detailed'")
        artifacts: Dict[str, Any] = {"openie": self.openie_results_path}

        stage = "OpenIE"
        try:
            # OpenIE is intentionally retained in both artifact modes.
            self.triple_extraction(docs)
            openie_payload = {"docs": self.openie_results or []}
            initial = ThreeLayerMemory()
            initial.build_from_raw_openie_results(openie_payload)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "initial_memory.json")
                initial.save(path)
                artifacts["initial_memory"] = path
                backup_path = os.path.join(self.global_config.save_dir, "backup", "initial_memory.backup.json")
                self._write_json(backup_path, initial.to_dict())
                artifacts["initial_memory_backup"] = backup_path

            stage = "schema extraction"
            with_schema = self.extract_memory_schema(initial)
            schema_path = os.path.join(self.global_config.save_dir, "initial_memory_with_schema.json")
            with_schema.save(schema_path)
            artifacts["schema_memory"] = schema_path

            stage = "ontology filtering"
            filtered = self.filter_memory_ontology(with_schema)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "memory_after_ontology_filtering.json")
                filtered.save(path)
                artifacts["filtered_memory"] = path

            stage = "conflict detection"
            conflicts = self.detect_memory_conflicts(filtered)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "conflict_detection_results.json")
                self._write_json(path, conflicts)
                artifacts["conflict_detection"] = path

            stage = "conflict resolution"
            final_memory, resolution = self.resolve_memory_conflicts(filtered, conflicts)
            if detailed:
                path = os.path.join(self.global_config.save_dir, "conflict_resolution_results.json")
                self._write_json(path, resolution)
                artifacts["conflict_resolution"] = path

            final_path = os.path.join(self.global_config.save_dir, "memory.json")
            final_memory.save(final_path)
            artifacts["memory"] = final_path
            self.memory = final_memory
            self.memory_path = final_path

            stage = "memory graph construction"
            memory_graph = self.build_memory_graph(final_memory)
            self.install_memory_graph(memory_graph)
            artifacts["memory_graph"] = self.save_memory_graph(memory_graph)
            artifacts["retrieval_graph"] = self._graphml_xml_file
            return {"memory": final_memory, "graph": memory_graph, "stats": memory_graph["stats"],
                    "conflict_summary": conflicts["summary"],
                    "resolution_summary": resolution["summary"], "artifacts": artifacts}
        except Exception as exc:
            raise RuntimeError(f"index_with_memory failed during {stage}: {exc}") from exc

    # ------------------------------------------------------------------
    #  Legacy HippoRAG-style indexing (flat graph, no type nodes)
    #  With no type nodes the NoN has a single "unassigned" domain, so
    #  CrossRank reduces to manifold ranking on the flat graph.
    # ------------------------------------------------------------------
    def index(self, docs: List[str]):
        """Index documents into a flat passage/entity graph (HippoRAG style)."""
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunks = self.chunk_embedding_store.get_text_for_all_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        self.openie_results = all_openie_info
        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunks) == len(ner_results_dict) == len(triple_results_dict)

        chunk_ids = list(chunks.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings(list(facts.values()))

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_num_chunk = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()

            self.augment_graph()
            self.save_igraph()
            self.ready_to_retrieve = False

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """Add entity-entity edges from triples and count passages per entity."""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_num_chunk[node] = self.ent_node_to_num_chunk.get(node, 0) + 1

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """Connect passage nodes to their entity nodes. Returns the number of new passages."""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info(f"Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):

            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")

                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """Add synonymy edges between similar entity nodes (KNN on entity embeddings)."""
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_text_for_all_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        self.node_to_node_stats[(node_key, nn)] = score
                        num_nns += 1

    # ------------------------------------------------------------------
    #  OpenIE bookkeeping
    # ------------------------------------------------------------------
    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """Load existing OpenIE results (if any) and return the chunk keys still to process."""
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            with open(self.openie_results_path) as f:
                openie_results = json.load(f)
            all_openie_info = openie_results.get('docs', [])

            # Standardizing indices for OpenIE files.
            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """Merge NER and triple extraction results with their passages into `all_openie_info`."""
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            triple_records = []
            seen_fact_ids = set()
            for triple in triple_results_dict[chunk_key].triples:
                record = normalize_triple_entry(triple)
                if record is None:
                    continue
                fact_id = record['fact_id']
                if fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact_id)
                triple_records.append({
                    'fact_id': fact_id,
                    'processed_triple': record['triple'],
                    'triple_str': record['triple_str'],
                    'raw_triple': triple
                })

            chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_records}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """Save OpenIE results with entity length statistics."""
        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            openie_dict = {'docs': all_openie_info, 'avg_ent_chars': round(sum_phrase_chars / max(num_phrases, 1), 4),
                           'avg_ent_words': round(sum_phrase_words / max(num_phrases, 1), 4)}
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    # ------------------------------------------------------------------
    #  Graph helpers (legacy path)
    # ------------------------------------------------------------------
    def augment_graph(self):
        """Add new nodes and edges to the graph."""
        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """Add entity and passage nodes that are not yet in the graph."""
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_nodes = self.entity_embedding_store.get_text_for_all_rows()
        passage_nodes = self.chunk_embedding_store.get_text_for_all_rows()

        nodes = entity_nodes
        nodes.update(passage_nodes)

        new_nodes = {}
        for node_id, node in nodes.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """Add edges from `node_to_node_stats`, skipping edges whose endpoints are not in the graph."""
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                valid_weights["weight"].append(edge_d.get("weight", 1.0))
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(
            valid_edges,
            attributes=valid_weights
        )

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_graphml(self._graphml_xml_file)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """Node / triple statistics of the legacy graph."""
        graph_info = {}

        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        passage_nodes_keys = self.chunk_embedding_store.get_all_ids()
        graph_info["num_passage_nodes"] = len(set(passage_nodes_keys))

        graph_info["num_total_nodes"] = graph_info["num_phrase_nodes"] + graph_info["num_passage_nodes"]

        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        passage_nodes_set = set(passage_nodes_keys)
        num_triples_with_passage_node = sum(
            1 for node_pair in self.node_to_node_stats
            if node_pair[0] in passage_nodes_set or node_pair[1] in passage_nodes_set
        )
        graph_info['num_triples_with_passage_node'] = num_triples_with_passage_node

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info[
            "num_extracted_triples"] - num_triples_with_passage_node

        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    # ------------------------------------------------------------------
    #  Entity -> triples lookup
    # ------------------------------------------------------------------
    def get_triples_by_entity(self, entity: str) -> List[Dict[str, Any]]:
        """All triples (with chunk id, passage, fact id) that contain `entity`."""
        if not self.entity_to_triples:
            self.build_entity_to_triples_index()

        processed_entity = text_processing(entity)
        results = self.entity_to_triples.get(processed_entity, [])

        logger.info(f"Found {len(results)} triples containing entity '{entity}'")
        return results

    def build_entity_to_triples_index(self):
        """Build self.entity_to_triples: processed entity -> list of {chunk_id, triple, passage, fact_id}."""
        logger.info("Building entity to triples index for fast lookup...")

        self.entity_to_triples = {}

        if not os.path.isfile(self.openie_results_path):
            logger.warning(f"OpenIE results file not found: {self.openie_results_path}")
            return

        with open(self.openie_results_path, 'r') as f:
            openie_results = json.load(f)

        all_openie_info = openie_results.get('docs', [])
        if not all_openie_info:
            logger.warning("No OpenIE results found in the file")
            return

        renamed_openie_info = []
        for openie_info in all_openie_info:
            openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
            renamed_openie_info.append(openie_info)

        all_openie_info = renamed_openie_info

        total_triples = 0
        for chunk_info in all_openie_info:
            chunk_id = chunk_info['idx']
            passage = chunk_info.get('passage', '')
            triple_entries = chunk_info.get('extracted_triples', [])

            triple_records = []
            seen_triples = set()
            for entry in triple_entries:
                record = normalize_triple_entry(entry)
                if record is None:
                    continue
                triple_tuple = tuple(record['triple'])
                if triple_tuple in seen_triples:
                    continue
                seen_triples.add(triple_tuple)
                triple_records.append(record)

            for record in triple_records:
                triple = record['triple']
                fact_id = record['fact_id']
                processed_subject = triple[0]
                processed_object = triple[2]

                triple_info = {
                    'chunk_id': chunk_id,
                    'triple': triple,
                    'passage': passage,
                    'fact_id': fact_id
                }

                total_triples += 1

                if processed_subject:
                    self.entity_to_triples.setdefault(processed_subject, []).append(triple_info)

                if processed_object and processed_object != processed_subject:
                    self.entity_to_triples.setdefault(processed_object, []).append(triple_info)

        logger.info(f"Entity index built: {len(self.entity_to_triples)} unique entities, {total_triples} total triples indexed")

    # ------------------------------------------------------------------
    #  Retrieval preparation, including the Network of Networks
    # ------------------------------------------------------------------
    def prepare_retrieval_objects(self):
        """Prepare in-memory objects for fast retrieval (embeddings, node keys, NoN ranker)."""

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        graph_node_names = set(self.graph.vs["name"])
        self.entity_node_keys: List = [key for key in self.entity_embedding_store.get_all_ids()
                                       if key in graph_node_names]
        self.passage_node_keys: List = [key for key in self.chunk_embedding_store.get_all_ids()
                                        if key in graph_node_names]
        self.type_node_keys: List = [node["name"] for node in self.graph.vs
                                     if node.attributes().get("layer") == "type"]
        all_fact_node_keys = set(self.fact_embedding_store.get_all_ids())
        final_memory_path = os.path.join(self.global_config.save_dir, "memory.json")
        if os.path.isfile(final_memory_path):
            final_memory = ThreeLayerMemory.load(final_memory_path)
            final_fact_keys = [
                compute_mdhash_id(str(tuple(map(str, fact.content))), prefix="fact-")
                for fact in final_memory.fact_layer
            ]
            self.fact_node_keys = [key for key in final_fact_keys if key in all_fact_node_keys]
        else:
            self.fact_node_keys = list(all_fact_node_keys)

        assert len(self.entity_node_keys) + len(self.passage_node_keys) + len(self.type_node_keys) == self.graph.vcount()

        igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}  # node key -> vertex index
        self.node_name_to_vertex_idx = igraph_name_to_idx
        self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
        self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys]

        entity_keys = set(self.entity_node_keys)
        passage_keys = set(self.passage_node_keys)
        entity_passages: Dict[str, Set[str]] = defaultdict(set)
        self.node_to_node_stats = {}
        for edge in self.graph.es:
            source = self.graph.vs[edge.source]["name"]
            target = self.graph.vs[edge.target]["name"]
            weight = float(edge["weight"]) if "weight" in edge.attributes() else 1.0
            self.node_to_node_stats[(source, target)] = weight
            if source in passage_keys and target in entity_keys:
                entity_passages[target].add(source)
            elif target in passage_keys and source in entity_keys:
                entity_passages[source].add(target)
        self.ent_node_to_num_chunk = {
            entity: len(entity_passages.get(entity, set())) for entity in entity_keys
        }

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        # Build entity to triples index for fast lookup
        self.build_entity_to_triples_index()

        # Build the Network of Networks and run CrossRank precomputation once
        self.build_non()

        self.ready_to_retrieve = True

    def build_non(self):
        """
        Build R = <G, A, theta> from the installed graph and precompute Anorm / Ynorm.

        main network G   : type nodes, weights from type_relation edges
        domain network A : entities of a type + passages mentioning them
        theta            : entity_to_type edges
        Nodes without a type go to one extra domain (the legacy flat graph has only that one).
        """
        g = self.graph
        n = g.vcount()

        layer_of = np.full(n, -1, dtype=np.int64)
        layer_of[np.array(self.passage_node_idxs, dtype=np.int64)] = PASSAGE
        layer_of[np.array(self.entity_node_idxs, dtype=np.int64)] = ENTITY
        type_idxs = np.array([self.node_name_to_vertex_idx[k] for k in self.type_node_keys], dtype=np.int64)
        layer_of[type_idxs] = TYPE

        edge_list = g.get_edgelist()
        if "weight" in g.es.attributes():
            edge_weights = [float(w) for w in g.es["weight"]]
        else:
            edge_weights = [1.0] * len(edge_list)

        if edge_list:
            rows = np.array([e[0] for e in edge_list], dtype=np.int64)
            cols = np.array([e[1] for e in edge_list], dtype=np.int64)
            W = sp.coo_matrix((np.array(edge_weights), (rows, cols)), shape=(n, n)).tocsr()
            W = W.maximum(W.T).tocsr()                   # undirected, as in directed=False
        else:
            W = sp.csr_matrix((n, n))

        alpha = getattr(self.global_config, "non_alpha", 1.0)
        c = getattr(self.global_config, "non_c", 0.5)
        logger.info(f"Building Network of Networks (alpha={alpha}, c={c}).")
        self.non = NoNCrossRank.from_layered_graph(W, layer_of, edge_list, edge_weights,
                                                   alpha=alpha, c=c)
        logger.info(f"NoN ready: {self.non.num_domains} domains, {len(self.non.node_of)} (domain, node) slots.")

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """Encode queries for fact and passage retrieval and cache them in `query_to_embedding`."""

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str, normalize: bool = True) -> np.ndarray:
        """Similarity between the query and every fact embedding (min-max normalized if requested)."""
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)  # shape: (#facts, )
        query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores

        if normalize:
            query_fact_scores = min_max_normalize(query_fact_scores)

        return query_fact_scores

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """Dense retrieval: returns passage indices sorted by normalized similarity, and the scores."""
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Keep only the weights of the top `link_top_k` phrases; zero the rest."""
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the query vector from the selected facts (entities) and dense passage scores,
        then rank passages with CrossRank on the Network of Networks.

        Returns (passage indices sorted by score, their scores).
        """
        linking_score_map = {}  # phrase -> average score of the facts that contain it
        phrase_scores = {}      # all fact scores per phrase, whether or not the phrase is in the graph
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            predicate_phrase = f[1].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores
            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    phrase_weights[phrase_id] = fact_score

                    if self.ent_node_to_num_chunk.get(phrase_key, 0) != 0:
                        phrase_weights[phrase_id] /= self.ent_node_to_num_chunk[phrase_key]

                if phrase not in phrase_scores:
                    phrase_scores[phrase] = []
                phrase_scores[phrase].append(fact_score)

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(link_top_k,
                                                                       phrase_weights,
                                                                       linking_score_map)

        # Passage scores from dense retrieval
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # Query vector e: phrase and passage seeds
        node_weights = phrase_weights + passage_weights

        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # CrossRank over the Network of Networks (replaces Personalized PageRank)
        ranked_doc_ids, ranked_doc_scores = self.run_crossrank(node_weights)

        assert len(ranked_doc_ids) == len(
            self.passage_node_idxs), f"Doc prob length {len(ranked_doc_ids)} != corpus length {len(self.passage_node_idxs)}"

        return ranked_doc_ids, ranked_doc_scores

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """
        Select the relevant facts for a query.

        Returns:
            top_k_fact_indices, top_k_facts, rerank_log
            rerank_log = {'facts_before_rerank': ..., 'facts_after_rerank': ...}
        """
        link_top_k: int = self.global_config.linking_top_k

        # skip_fact_rerank + use_raw_threshold_filter: threshold ALL facts on raw similarity (no top-k)
        if self.global_config.skip_fact_rerank and self.global_config.use_raw_threshold_filter:
            threshold = self.global_config.fact_similarity_threshold
            raw_fact_scores = self.get_fact_scores(query, normalize=False)

            all_fact_indices = list(range(len(query_fact_scores)))
            fact_row_dict = self.fact_embedding_store.get_rows(self.fact_node_keys)

            filtered_indices = []
            filtered_facts = []
            candidate_scores = []

            for idx in all_fact_indices:
                raw_score = raw_fact_scores[idx]
                candidate_scores.append(raw_score)
                if raw_score >= threshold:
                    filtered_indices.append(idx)
                    fact_content = fact_row_dict[self.fact_node_keys[idx]]['content']
                    filtered_facts.append(ast.literal_eval(fact_content))

            rerank_log = {
                'facts_before_rerank': len(all_fact_indices),
                'facts_after_rerank': filtered_facts,
                'rerank_skipped': True,
                'use_raw_threshold_filter': True,
                'similarity_threshold': threshold,
                'filtered_count': len(filtered_facts),
                'max_candidate_score': float(max(candidate_scores)),
                'min_candidate_score': float(min(candidate_scores))
            }
            return filtered_indices, filtered_facts, rerank_log

        # Original logic: take top-k by score, then filter
        candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
        real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
        fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [ast.literal_eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]

        # skip_fact_rerank: filter by raw similarity threshold instead of the LLM reranker
        if self.global_config.skip_fact_rerank:
            threshold = self.global_config.fact_similarity_threshold
            raw_fact_scores = self.get_fact_scores(query, normalize=False)

            filtered_indices = []
            filtered_facts = []
            candidate_scores = []
            for idx, fact in zip(candidate_fact_indices, candidate_facts):
                raw_score = raw_fact_scores[idx]
                candidate_scores.append(raw_score)
                if raw_score >= threshold:
                    filtered_indices.append(idx)
                    filtered_facts.append(fact)

            rerank_log = {
                'facts_before_rerank': candidate_facts,
                'facts_after_rerank': filtered_facts,
                'rerank_skipped': True,
                'similarity_threshold': threshold,
                'filtered_count': len(filtered_facts),
                'max_candidate_score': float(max(candidate_scores)),
                'min_candidate_score': float(min(candidate_scores))
            }
            return filtered_indices, filtered_facts, rerank_log

        top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                             candidate_facts,
                                                                             candidate_fact_indices,
                                                                             len_after_rerank=link_top_k)

        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}

        return top_k_fact_indices, top_k_facts, rerank_log

    def run_crossrank(self, query_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rank passage nodes with CrossRank on the Network of Networks.

        query_vector: (n,) seed vector over all graph nodes (NaN / negative values are zeroed).
        Returns (passage indices sorted by score descending, their scores).
        """
        scores = self.non.run(query_vector,
                              max_iter=getattr(self.global_config, "non_max_iter", 100),
                              epsilon=getattr(self.global_config, "non_epsilon", 1e-10))

        doc_scores = np.array([scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores

    # ------------------------------------------------------------------
    #  Retrieval and QA
    # ------------------------------------------------------------------
    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Retrieval with MemGraphRAG:
        - Fact retrieval and selection (rerank or similarity threshold)
        - Dense passage scoring
        - CrossRank on the Network of Networks (instead of Personalized PageRank)

        Queries with no relevant facts fall back to dense passage retrieval.
        Returns the QuerySolution list (plus overall retrieval metrics if `gold_docs` is given).
        """

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        # Thread-safe counters
        fact_stats = {'no_facts': 0, 'has_facts': 0, 'total_facts_before': 0, 'total_facts_after': 0}
        fact_stats_lock = threading.Lock()

        def retrieve_single_query(query):
            """Retrieve for one query."""
            # Always use normalized scores for the graph search
            query_fact_scores = self.get_fact_scores(query, normalize=True)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)

            facts_before = rerank_log.get('facts_before_rerank', [])
            if not isinstance(facts_before, int):
                facts_before = len(facts_before)
            facts_after = len(rerank_log.get('facts_after_rerank', []))

            if len(top_k_facts) == 0:
                with fact_stats_lock:
                    fact_stats['no_facts'] += 1
                    fact_stats['total_facts_before'] += facts_before
                    fact_stats['total_facts_after'] += facts_after
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                with fact_stats_lock:
                    fact_stats['has_facts'] += 1
                    fact_stats['total_facts_before'] += facts_before
                    fact_stats['total_facts_after'] += facts_after
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         passage_node_weight=self.global_config.passage_node_weight)

            top_k_docs = [self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]
            return QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve])

        retrieval_max_workers = self.global_config.retrieval_max_workers
        if retrieval_max_workers > 1:
            retrieval_results = [None] * len(queries)
            with ThreadPoolExecutor(max_workers=retrieval_max_workers) as executor:
                future_to_idx = {executor.submit(retrieve_single_query, query): idx
                                 for idx, query in enumerate(queries)}
                for future in tqdm(as_completed(future_to_idx), total=len(queries), desc="Retrieving (parallel)"):
                    idx = future_to_idx[future]
                    retrieval_results[idx] = future.result()
        else:
            retrieval_results = []
            for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
                retrieval_results.append(retrieve_single_query(query))

        # Fact statistics log
        total_queries = max(fact_stats['no_facts'] + fact_stats['has_facts'], 1)
        total_facts_before = fact_stats['total_facts_before']
        total_facts_after = fact_stats['total_facts_after']
        total_facts_filtered = total_facts_before - total_facts_after
        filter_rate = (total_facts_filtered / total_facts_before * 100) if total_facts_before > 0 else 0
        logger.info(f"Fact reranking stats: total_queries={total_queries}, "
                    f"has_facts={fact_stats['has_facts']} ({fact_stats['has_facts']/total_queries*100:.1f}%), "
                    f"no_facts={fact_stats['no_facts']} ({fact_stats['no_facts']/total_queries*100:.1f}%)")
        logger.info(f"Fact filtering stats: total_before={total_facts_before}, total_after={total_facts_after}, "
                    f"filtered={total_facts_filtered} ({filter_rate:.1f}%), "
                    f"avg_per_query_before={total_facts_before/total_queries:.1f}, avg_per_query_after={total_facts_after/total_queries:.1f}")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Retrieval-augmented QA. Queries may be strings (retrieval is run) or QuerySolution objects.

        Returns (solutions, response messages, metadata, [retrieval metrics, QA metrics,] retrieval_time).
        Retrieval and QA metrics are included only when `gold_answers` is given.
        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        overall_retrieval_result = None
        retrieval_time = 0

        if not isinstance(queries[0], QuerySolution):
            retrieval_start = time.time()
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)
            retrieval_time = time.time() - retrieval_start
            logger.info(f"Retrieval time: {retrieval_time:.2f}s")

        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results, retrieval_time
        else:
            return queries_solutions, all_response_message, all_metadata, retrieval_time

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """Run QA inference over the retrieved passages of each QuerySolution."""
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                prompt_dataset_name = self.global_config.dataset
            else:
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))

        # Save the prompts of the first 5 questions
        sample_prompts_path = os.path.join(self.working_dir, "sample_qa_prompts.json")
        sample_prompts = []
        for i, qa_messages in enumerate(all_qa_messages[:5]):
            sample_prompts.append({
                "index": i,
                "question": queries[i].question,
                "messages": qa_messages
            })
        with open(sample_prompts_path, 'w', encoding='utf-8') as f:
            json.dump(sample_prompts, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(sample_prompts)} sample QA prompts to {sample_prompts_path}")

        qa_max_workers = self.global_config.qa_max_workers
        if qa_max_workers > 1:
            all_qa_results = [None] * len(all_qa_messages)
            with ThreadPoolExecutor(max_workers=qa_max_workers) as executor:
                future_to_idx = {executor.submit(self.llm_model.infer, qa_messages): idx
                                 for idx, qa_messages in enumerate(all_qa_messages)}
                for future in tqdm(as_completed(future_to_idx), total=len(all_qa_messages), desc="QA Reading (parallel)"):
                    idx = future_to_idx[future]
                    all_qa_results[idx] = future.result()
        else:
            all_qa_results = [self.llm_model.infer(qa_messages) for qa_messages in tqdm(all_qa_messages, desc="QA Reading")]

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        # Extract predicted answers
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata
