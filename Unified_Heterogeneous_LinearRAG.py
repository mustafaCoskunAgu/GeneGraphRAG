"""
Unified heterogeneous LinearRAG.

Graph node types:
    P: passages
    S: sentences
    E: entities

Query-time retrieval uses one query-dependent Personalized PageRank (PPR)
over the heterogeneous P-S-E graph.

Expected project dependencies:
    - src.embedding_store.EmbeddingStore
    - src.ner.SpacyNER
    - numpy
    - scipy
    - tqdm

This implementation intentionally does not use igraph or torch. The
heterogeneous adjacency and PPR iterations are performed with SciPy sparse
matrices.
"""

import os
import json
import re
import math
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from tqdm import tqdm

from src.embedding_store import EmbeddingStore
from src.ner import SpacyNER


logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(self, global_config):
        self.config = global_config
        self.dataset_name = global_config.dataset_name
        self.llm_model = global_config.llm_model

        self.spacy_ner = SpacyNER(global_config.spacy_model)
        self.load_embedding_store()

        self.ner_results_path = os.path.join(
            self.config.working_dir, self.dataset_name, "ner_results.json"
        )

        # Populated by index().
        self.passage_hash_ids = []
        self.sentence_hash_ids = []
        self.entity_hash_ids = []

        self.passage_hash_id_to_idx = {}
        self.sentence_hash_id_to_idx = {}
        self.entity_hash_id_to_idx = {}

        self.passage_hash_id_to_entities = {}
        self.sentence_hash_id_to_entity_hash_ids = {}
        self.entity_hash_id_to_sentence_hash_ids = {}

        # Sparse heterogeneous graph edge lists.
        self.ps_passage_indices = []
        self.ps_sentence_indices = []
        self.pe_passage_indices = []
        self.pe_entity_indices = []
        self.se_sentence_indices = []
        self.se_entity_indices = []
        self.se_base_weights = []

        self.adjacent_passage_edges = []

        logger.info("Initialized unified heterogeneous LinearRAG.")

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _cfg(self, name, default):
        """Read a config value while remaining compatible with older configs."""
        return getattr(self.config, name, default)

    def load_embedding_store(self):
        dataset_dir = os.path.join(
            self.config.working_dir, self.dataset_name
        )
        os.makedirs(dataset_dir, exist_ok=True)

        self.passage_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(dataset_dir, "passage_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="passage",
        )
        self.entity_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(dataset_dir, "entity_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="entity",
        )
        self.sentence_embedding_store = EmbeddingStore(
            self.config.embedding_model,
            db_filename=os.path.join(dataset_dir, "sentence_embedding.parquet"),
            batch_size=self.config.batch_size,
            namespace="sentence",
        )

    @staticmethod
    def _ordered_hash_ids(store):
        """
        Return IDs in embedding-row order.

        EmbeddingStore versions commonly expose hash_ids. The fallback uses
        hash_id_to_idx to recover the row order.
        """
        ids = getattr(store, "hash_ids", None)
        if ids is not None and len(ids) == len(store.embeddings):
            return list(ids)

        mapping = store.hash_id_to_idx
        return [
            hash_id
            for hash_id, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]

    @staticmethod
    def _normalize_nonnegative(values):
        """Convert scores to a finite, nonnegative probability vector."""
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        values = np.maximum(values, 0.0)
        total = values.sum()

        if total > 0:
            return values / total
        return np.zeros_like(values)

    @staticmethod
    def _normalise_text(text):
        return re.sub(r"\s+", " ", str(text)).strip().lower()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def load_existing_data(self, passage_hash_ids):
        if os.path.exists(self.ner_results_path):
            with open(self.ner_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            passage_entities = data.get("passage_hash_id_to_entities", {})
            sentence_entities = data.get("sentence_to_entities", {})
            existing_ids = set(passage_entities.keys())
            new_ids = set(passage_hash_ids) - existing_ids

            return passage_entities, sentence_entities, new_ids

        return {}, {}, set(passage_hash_ids)

    def index(self, passages):
        """
        Index passages, extract entities/sentences, and construct the
        heterogeneous P-S-E edge lists.

        The passage-sentence mapping is recovered by matching sentence text
        produced by NER against sentence-like spans in each source passage.
        """
        logger.info("Indexing %d passages.", len(passages))

        self.passage_embedding_store.insert_text(passages)
        hash_id_to_passage = self.passage_embedding_store.get_hash_id_to_text()

        (
            existing_passage_hash_id_to_entities,
            existing_sentence_to_entities,
            new_passage_hash_ids,
        ) = self.load_existing_data(hash_id_to_passage.keys())

        if new_passage_hash_ids:
            new_hash_id_to_passage = {
                key: hash_id_to_passage[key] for key in new_passage_hash_ids
            }

            new_passage_entities, new_sentence_entities = (
                self.spacy_ner.batch_ner(
                    new_hash_id_to_passage, self.config.max_workers
                )
            )

            existing_passage_hash_id_to_entities.update(new_passage_entities)
            existing_sentence_to_entities.update(new_sentence_entities)

        os.makedirs(os.path.dirname(self.ner_results_path), exist_ok=True)
        with open(self.ner_results_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "passage_hash_id_to_entities":
                        existing_passage_hash_id_to_entities,
                    "sentence_to_entities": existing_sentence_to_entities,
                },
                f,
                ensure_ascii=False,
            )

        # Insert all unique sentence and entity texts into their stores.
        entity_texts = set()
        for entities in existing_passage_hash_id_to_entities.values():
            entity_texts.update(entities)

        sentence_texts = set(existing_sentence_to_entities.keys())

        if sentence_texts:
            self.sentence_embedding_store.insert_text(list(sentence_texts))
        if entity_texts:
            self.entity_embedding_store.insert_text(list(entity_texts))

        self.passage_hash_id_to_entities = (
            existing_passage_hash_id_to_entities
        )

        self._prepare_node_indices()
        self._prepare_entity_sentence_edges(existing_sentence_to_entities)
        self._prepare_passage_entity_edges()
        self._prepare_passage_sentence_edges(hash_id_to_passage)
        self._prepare_adjacent_passage_edges(hash_id_to_passage)

        logger.info(
            "Heterogeneous graph prepared: %d passages, %d sentences, "
            "%d entities; P-S=%d, P-E=%d, S-E=%d edges.",
            len(self.passage_hash_ids),
            len(self.sentence_hash_ids),
            len(self.entity_hash_ids),
            len(self.ps_passage_indices),
            len(self.pe_passage_indices),
            len(self.se_sentence_indices),
        )

    def _prepare_node_indices(self):
        self.passage_hash_ids = self._ordered_hash_ids(
            self.passage_embedding_store
        )
        self.sentence_hash_ids = self._ordered_hash_ids(
            self.sentence_embedding_store
        )
        self.entity_hash_ids = self._ordered_hash_ids(
            self.entity_embedding_store
        )

        self.passage_hash_id_to_idx = {
            hash_id: i for i, hash_id in enumerate(self.passage_hash_ids)
        }
        self.sentence_hash_id_to_idx = {
            hash_id: i for i, hash_id in enumerate(self.sentence_hash_ids)
        }
        self.entity_hash_id_to_idx = {
            hash_id: i for i, hash_id in enumerate(self.entity_hash_ids)
        }

        self.passage_embeddings = np.asarray(
            self.passage_embedding_store.embeddings, dtype=np.float32
        )
        self.sentence_embeddings = np.asarray(
            self.sentence_embedding_store.embeddings, dtype=np.float32
        )
        self.entity_embeddings = np.asarray(
            self.entity_embedding_store.embeddings, dtype=np.float32
        )

    def _prepare_entity_sentence_edges(self, sentence_to_entities):
        self.se_sentence_indices = []
        self.se_entity_indices = []
        self.se_base_weights = []

        self.sentence_hash_id_to_entity_hash_ids = {}
        self.entity_hash_id_to_sentence_hash_ids = defaultdict(list)

        for sentence_text, entities in sentence_to_entities.items():
            sentence_hash_id = (
                self.sentence_embedding_store.text_to_hash_id.get(sentence_text)
            )
            if sentence_hash_id is None:
                continue

            sentence_idx = self.sentence_hash_id_to_idx.get(sentence_hash_id)
            if sentence_idx is None:
                continue

            entity_hash_ids = []
            for entity_text in entities:
                entity_hash_id = (
                    self.entity_embedding_store.text_to_hash_id.get(entity_text)
                )
                if entity_hash_id is None:
                    continue

                entity_idx = self.entity_hash_id_to_idx.get(entity_hash_id)
                if entity_idx is None:
                    continue

                entity_hash_ids.append(entity_hash_id)
                self.se_sentence_indices.append(sentence_idx)
                self.se_entity_indices.append(entity_idx)
                self.se_base_weights.append(1.0)

                self.entity_hash_id_to_sentence_hash_ids[
                    entity_hash_id
                ].append(sentence_hash_id)

            self.sentence_hash_id_to_entity_hash_ids[
                sentence_hash_id
            ] = entity_hash_ids

        self.se_base_weights = np.asarray(
            self.se_base_weights, dtype=np.float32
        )

    def _prepare_passage_entity_edges(self):
        self.pe_passage_indices = []
        self.pe_entity_indices = []
        self.pe_base_weights = []

        for passage_hash_id, entities in self.passage_hash_id_to_entities.items():
            passage_idx = self.passage_hash_id_to_idx.get(passage_hash_id)
            if passage_idx is None:
                continue

            passage_text = self.passage_embedding_store.hash_id_to_text.get(
                passage_hash_id, ""
            )
            counts = []

            for entity_text in entities:
                entity_hash_id = (
                    self.entity_embedding_store.text_to_hash_id.get(entity_text)
                )
                if entity_hash_id is None:
                    continue

                entity_idx = self.entity_hash_id_to_idx.get(entity_hash_id)
                if entity_idx is None:
                    continue

                # Preserve the original implementation's mention-count idea.
                count = passage_text.lower().count(str(entity_text).lower())
                counts.append((entity_idx, max(count, 1)))

            total_count = sum(count for _, count in counts)
            if total_count == 0:
                continue

            for entity_idx, count in counts:
                self.pe_passage_indices.append(passage_idx)
                self.pe_entity_indices.append(entity_idx)
                self.pe_base_weights.append(float(count) / float(total_count))

    def _prepare_passage_sentence_edges(self, hash_id_to_passage):
        """
        Build B (passage-sentence incidence).

        NER output in the supplied implementation stores sentence->entities,
        but not passage->sentences. We therefore split each passage into
        sentence-like spans and match them to the sentence texts returned by
        NER. Exact normalized matches are used.
        """
        self.ps_passage_indices = []
        self.ps_sentence_indices = []

        normalized_sentence_to_idx = {}
        for sentence_hash_id in self.sentence_hash_ids:
            sentence = self.sentence_embedding_store.hash_id_to_text[
                sentence_hash_id
            ]
            normalized_sentence_to_idx[
                self._normalise_text(sentence)
            ] = self.sentence_hash_id_to_idx[sentence_hash_id]

        unmatched = 0
        for passage_hash_id, passage_text in hash_id_to_passage.items():
            passage_idx = self.passage_hash_id_to_idx.get(passage_hash_id)
            if passage_idx is None:
                continue

            # A practical fallback sentence splitter. NER-produced sentence
            # strings are matched after whitespace normalization.
            spans = re.split(r"(?<=[.!?])\s+|\n+", str(passage_text))
            matched_here = set()

            for span in spans:
                normalized = self._normalise_text(span)
                sentence_idx = normalized_sentence_to_idx.get(normalized)
                if sentence_idx is not None:
                    matched_here.add(sentence_idx)

            # If a sentence is not matched by splitting, attempt normalized
            # substring matching. This is slower, so it is only used for
            # passages with no exact matches.
            if not matched_here:
                normalized_passage = self._normalise_text(passage_text)
                for normalized_sentence, sentence_idx in (
                    normalized_sentence_to_idx.items()
                ):
                    if normalized_sentence and normalized_sentence in normalized_passage:
                        matched_here.add(sentence_idx)

            if not matched_here:
                unmatched += 1

            for sentence_idx in matched_here:
                self.ps_passage_indices.append(passage_idx)
                self.ps_sentence_indices.append(sentence_idx)

        if unmatched:
            logger.warning(
                "%d passages had no matched sentence nodes. Their P-E edges "
                "remain available, but they will not have P-S connections. "
                "Consider storing passage-to-sentence mappings directly "
                "during NER indexing.",
                unmatched,
            )

    def _prepare_adjacent_passage_edges(self, hash_id_to_passage):
        self.adjacent_passage_edges = []

        if not self._cfg("use_adjacent_passage_edges", True):
            return

        pattern = re.compile(r"^(\d+):")
        indexed = []

        for hash_id, text in hash_id_to_passage.items():
            match = pattern.match(str(text).strip())
            if match:
                idx = self.passage_hash_id_to_idx.get(hash_id)
                if idx is not None:
                    indexed.append((int(match.group(1)), idx))

        indexed.sort(key=lambda item: item[0])
        self.adjacent_passage_edges = [
            (indexed[i][1], indexed[i + 1][1])
            for i in range(len(indexed) - 1)
        ]

    # ------------------------------------------------------------------
    # Query initialization
    # ------------------------------------------------------------------

    def get_seed_entities(self, question):
        question_entities = list(self.spacy_ner.question_ner(question))
        if not question_entities or len(self.entity_embeddings) == 0:
            return np.zeros(len(self.entity_hash_ids), dtype=np.float32)

        query_entity_embeddings = self.config.embedding_model.encode(
            question_entities,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )
        query_entity_embeddings = np.asarray(
            query_entity_embeddings, dtype=np.float32
        )

        similarities = self.entity_embeddings @ query_entity_embeddings.T
        seed_scores = np.zeros(
            len(self.entity_hash_ids), dtype=np.float32
        )

        for query_entity_idx in range(similarities.shape[1]):
            best_idx = int(np.argmax(similarities[:, query_entity_idx]))
            best_score = float(similarities[best_idx, query_entity_idx])
            seed_scores[best_idx] = max(
                seed_scores[best_idx], max(best_score, 0.0)
            )

        return seed_scores

    def _query_similarity_scores(self, question_embedding):
        if len(self.passage_embeddings):
            passage_scores = (
                self.passage_embeddings @ question_embedding.reshape(-1)
            )
        else:
            passage_scores = np.zeros(0, dtype=np.float32)

        if len(self.sentence_embeddings):
            sentence_scores = (
                self.sentence_embeddings @ question_embedding.reshape(-1)
            )
        else:
            sentence_scores = np.zeros(0, dtype=np.float32)

        # Cosine similarities can be negative; random-walk weights cannot.
        passage_scores = np.maximum(passage_scores, 0.0)
        sentence_scores = np.maximum(sentence_scores, 0.0)

        return passage_scores, sentence_scores

    def _build_personalization_vector(
        self, passage_scores, sentence_scores, entity_scores
    ):
        n_p = len(self.passage_hash_ids)
        n_s = len(self.sentence_hash_ids)
        n_e = len(self.entity_hash_ids)

        p_p = self._normalize_nonnegative(passage_scores)
        p_s = self._normalize_nonnegative(sentence_scores)
        p_e = self._normalize_nonnegative(entity_scores)

        # Configurable mixture of query-to-node-type relevance.
        beta_p = float(self._cfg("ppr_passage_seed_weight", 0.20))
        beta_s = float(self._cfg("ppr_sentence_seed_weight", 0.30))
        beta_e = float(self._cfg("ppr_entity_seed_weight", 0.50))

        components = [
            (beta_p, p_p, 0, n_p),
            (beta_s, p_s, n_p, n_p + n_s),
            (beta_e, p_e, n_p + n_s, n_p + n_s + n_e),
        ]

        # If a node type has no usable query seed, redistribute its weight
        # across the node types that do have a nonzero personalization vector.
        available = [
            (weight, vector, start, end)
            for weight, vector, start, end in components
            if len(vector) > 0 and vector.sum() > 0 and weight > 0
        ]

        if not available:
            # Last-resort fallback: uniform over passages.
            if n_p == 0:
                raise RuntimeError("No passage nodes are available.")
            p = np.zeros(n_p + n_s + n_e, dtype=np.float64)
            p[:n_p] = 1.0 / n_p
            return p

        weight_sum = sum(weight for weight, _, _, _ in available)
        p = np.zeros(n_p + n_s + n_e, dtype=np.float64)

        for weight, vector, start, end in available:
            p[start:end] += (weight / weight_sum) * vector

        total = p.sum()
        if total > 0:
            p /= total
        return p

    # ------------------------------------------------------------------
    # Query-dependent heterogeneous graph and PPR
    # ------------------------------------------------------------------

    def _build_query_adjacency(self, sentence_scores):
        """
        Construct symmetric A_q in node order [P, S, E]:

            [ 0      B       C  ]
        A = [ B.T    0     DsigM]
            [ C.T  M.TDsig    0 ]

        The query-dependent sentence relevance weights the S-E edges.
        """
        n_p = len(self.passage_hash_ids)
        n_s = len(self.sentence_hash_ids)
        n_e = len(self.entity_hash_ids)
        n_total = n_p + n_s + n_e

        rows, cols, vals = [], [], []

        ps_weight = float(self._cfg("passage_sentence_edge_weight", 1.0))
        pe_weight = float(self._cfg("passage_entity_edge_weight", 1.0))
        se_weight = float(self._cfg("sentence_entity_edge_weight", 1.0))
        adjacent_weight = float(
            self._cfg("adjacent_passage_edge_weight", 0.10)
        )

        # Scale query sentence relevance to [0, 1] without making all
        # corpus-wide similarities sum to one (which can become tiny).
        sigma = np.asarray(sentence_scores, dtype=np.float64)
        sigma = np.maximum(np.nan_to_num(sigma), 0.0)
        if sigma.size and sigma.max() > 0:
            sigma = sigma / sigma.max()

        # P <-> S edges (B and B.T).
        for p_idx, s_idx in zip(
            self.ps_passage_indices, self.ps_sentence_indices
        ):
            p_node = p_idx
            s_node = n_p + s_idx
            rows.extend([p_node, s_node])
            cols.extend([s_node, p_node])
            vals.extend([ps_weight, ps_weight])

        # P <-> E edges (C and C.T).
        # Count-normalized entity weights are reconstructed per passage.
        passage_entity_weights = defaultdict(float)
        for p_idx, e_idx, base_weight in zip(
            self.pe_passage_indices,
            self.pe_entity_indices,
            self.pe_base_weights,
        ):
            passage_entity_weights[(p_idx, e_idx)] += float(base_weight)

        for (p_idx, e_idx), base_weight in passage_entity_weights.items():
                p_node = p_idx
                e_node = n_p + n_s + e_idx
                weight = pe_weight * base_weight
                rows.extend([p_node, e_node])
                cols.extend([e_node, p_node])
                vals.extend([weight, weight])

        # S <-> E edges, weighted by query relevance sigma_s.
        for s_idx, e_idx in zip(
            self.se_sentence_indices, self.se_entity_indices
        ):
            weight = se_weight * float(sigma[s_idx])
            if weight <= 0:
                continue

            s_node = n_p + s_idx
            e_node = n_p + n_s + e_idx
            rows.extend([s_node, e_node])
            cols.extend([e_node, s_node])
            vals.extend([weight, weight])

        # Optional adjacent passage links retained from the original code.
        for p1, p2 in self.adjacent_passage_edges:
            if p1 == p2:
                continue
            rows.extend([p1, p2])
            cols.extend([p2, p1])
            vals.extend([adjacent_weight, adjacent_weight])

        adjacency = sparse.coo_matrix(
            (vals, (rows, cols)),
            shape=(n_total, n_total),
            dtype=np.float64,
        ).tocsr()

        adjacency.sum_duplicates()
        return adjacency

    @staticmethod
    def _run_ppr(adjacency, personalization, damping=0.85,
                 max_iter=100, tol=1e-8):
        """
        Solve x = (1-damping)*p + damping*T.T*x.

        Here damping is the probability of continuing the random walk.
        Dangling-node probability is redistributed according to p.
        """
        n = adjacency.shape[0]
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
        inv_degree = np.zeros_like(degree, dtype=np.float64)
        nonzero = degree > 0
        inv_degree[nonzero] = 1.0 / degree[nonzero]

        transition = sparse.diags(inv_degree) @ adjacency
        dangling = ~nonzero

        p = np.asarray(personalization, dtype=np.float64)
        p = np.maximum(np.nan_to_num(p), 0.0)
        p_sum = p.sum()
        if p_sum == 0:
            p = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            p /= p_sum

        d = float(damping)
        d = min(max(d, 0.0), 0.999999)
        restart = 1.0 - d

        x = p.copy()
        for iteration in range(int(max_iter)):
            dangling_mass = x[dangling].sum()
            x_new = (
                restart * p
                + d * (
                    transition.T.dot(x)
                    + dangling_mass * p
                )
            )

            delta = np.linalg.norm(x_new - x, ord=1)
            x = x_new

            if delta < tol:
                break

        return x

    def graph_search_with_seed_entities(
        self,
        question,
        question_embedding,
        seed_entity_scores=None,
    ):
        passage_scores, sentence_scores = (
            self._query_similarity_scores(question_embedding)
        )

        if seed_entity_scores is None:
            seed_entity_scores = self.get_seed_entities(question)

        personalization = self._build_personalization_vector(
            passage_scores,
            sentence_scores,
            seed_entity_scores,
        )

        adjacency = self._build_query_adjacency(sentence_scores)

        ppr_scores = self._run_ppr(
            adjacency=adjacency,
            personalization=personalization,
            damping=float(self._cfg("damping", 0.85)),
            max_iter=int(self._cfg("ppr_max_iter", 100)),
            tol=float(self._cfg("ppr_tolerance", 1e-8)),
        )

        n_p = len(self.passage_hash_ids)
        passage_ppr_scores = ppr_scores[:n_p]

        top_indices = np.argsort(passage_ppr_scores)[::-1]
        sorted_hash_ids = [
            self.passage_hash_ids[i] for i in top_indices
        ]
        sorted_scores = passage_ppr_scores[top_indices].tolist()

        return sorted_hash_ids, sorted_scores

    # ------------------------------------------------------------------
    # Retrieval and QA
    # ------------------------------------------------------------------

    def retrieve(self, questions):
        if not self.passage_hash_ids:
            raise RuntimeError(
                "No indexed graph is available. Call index(passages) first."
            )

        retrieval_results = []

        for question_info in tqdm(questions, desc="Unified PPR retrieval"):
            question = question_info["question"]

            question_embedding = self.config.embedding_model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=self.config.batch_size,
            )
            question_embedding = np.asarray(
                question_embedding, dtype=np.float32
            ).reshape(-1)

            seed_entity_scores = self.get_seed_entities(question)

            sorted_hash_ids, sorted_scores = (
                self.graph_search_with_seed_entities(
                    question,
                    question_embedding,
                    seed_entity_scores,
                )
            )

            top_k = int(self._cfg("retrieval_top_k", 5))
            final_hash_ids = sorted_hash_ids[:top_k]
            final_scores = sorted_scores[:top_k]

            final_passages = [
                self.passage_embedding_store.hash_id_to_text[hash_id]
                for hash_id in final_hash_ids
            ]

            result = {
                "question": question,
                "sorted_passage": final_passages,
                "sorted_passage_scores": final_scores,
                "gold_answer": question_info.get("answer"),
            }
            retrieval_results.append(result)

        return retrieval_results

    def qa(self, questions):
        retrieval_results = self.retrieve(questions)

        system_prompt = (
            "As an advanced reading comprehension assistant, analyze the "
            "provided passages and answer the question. Explain the reasoning "
            "briefly, then conclude with 'Answer: ' followed by a concise answer."
        )

        all_messages = []
        for result in retrieval_results:
            context = "\n\n".join(result["sorted_passage"])
            user_prompt = (
                f"Passages:\n{context}\n\n"
                f"Question: {result['question']}\n\n"
                "Reasoning: "
            )
            all_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])

        with ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            outputs = list(tqdm(
                executor.map(self.llm_model.infer, all_messages),
                total=len(all_messages),
                desc="QA reading (parallel)",
            ))

        for output, result in zip(outputs, retrieval_results):
            if "Answer:" in output:
                result["pred_answer"] = output.split("Answer:", 1)[1].strip()
            else:
                result["pred_answer"] = output.strip()

        return retrieval_results
