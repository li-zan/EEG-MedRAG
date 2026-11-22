"""Medical extraction, KHG/CHG/EVD construction and multilayer retrieval."""

from __future__ import annotations
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeAlias, cast

import asyncio
from copy import copy
import logging
import json
import math
import re
from time import perf_counter
from collections import defaultdict
from dataclasses import asdict, replace

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike
from numba import njit

from .base import BuildConfig, Case, Document, Evidence, QueryConfig, RetrievalCandidates, Segment, Query, Tokenizer
from .dataset import DatasetCorpus
from .storage import CachedModel, Storage
from .clinical import (
    attribute_matches, clinical_content, query_attributes,
)
from ._vendor.hypergraphrag import operate as native_operate
from ._vendor.hypergraphrag.prompt import GRAPH_FIELD_SEP
from .context import ContextBuilder
from .prompt import ENTITY_SUMMARY_INSTRUCTION, PROMPTS
from .utils import IntegrityError, canonical, identity

logger = logging.getLogger(__name__)

RankedNode: TypeAlias = tuple[dict, float, np.ndarray]


# Medical extraction and cached continuation rounds
def extraction_records(text: str) -> Iterator[list[str]]:
    """Split extraction output into delimited hyper-relation and entity records."""
    native = native_operate
    records = native.split_string_by_multi_markers(
        text,
        [PROMPTS["DEFAULT_RECORD_DELIMITER"], PROMPTS["DEFAULT_COMPLETION_DELIMITER"]],
    )
    for record in records:
        match = re.search(r"\((.*)\)", record)
        if match is not None:
            yield native.split_string_by_multi_markers(
                match.group(1), [PROMPTS["DEFAULT_TUPLE_DELIMITER"]]
            )


def validate_extraction(text: str, prefix: str = "") -> str:
    """Validate record structure, ordering, completion markers and score ranges."""
    if not (type(text) is str and text.strip()):
        raise IntegrityError("Empty extraction output")
    completion = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
    if not text.rstrip().endswith(completion):
        raise IntegrityError("Missing extraction completion delimiter")
    current = False
    records = list(extraction_records(prefix + text))
    if not records and text.strip() != completion:
        raise IntegrityError("Malformed extraction records")
    for fields in records:
        kind = fields[0]
        if kind == '"hyper-relation"':
            if len(fields) != 3 or not fields[1].strip(' "'):
                raise IntegrityError("Invalid hyper-relation record")
            maximum = 10
            current = True
        elif kind == '"entity"':
            if len(fields) != 5 or not all(v.strip(' "') for v in fields[1:4]):
                raise IntegrityError("Invalid entity record")
            if not current:
                raise IntegrityError("Entity record precedes its hyper-relation")
            maximum = 100
        else:
            raise IntegrityError("Unknown extraction record type")
        try:
            number = float(fields[-1])
        except ValueError as error:
            raise IntegrityError("Invalid extraction score") from error
        if not (math.isfinite(number) and 0 <= number <= maximum):
            raise IntegrityError("Invalid extraction score")
    return text


class Extractor:
    def __init__(self, model: CachedModel, config: BuildConfig | None = None) -> None:
        self.model = model
        self.config = config or BuildConfig()

    async def facts(self, document: Document) -> str:
        """Extract facts as validated tuple-delimited records."""
        context = {
            "tuple_delimiter": PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            "record_delimiter": PROMPTS["DEFAULT_RECORD_DELIMITER"],
            "completion_delimiter": PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            "language": PROMPTS["DEFAULT_LANGUAGE"],
        }
        examples = "\n".join(PROMPTS["entity_extraction_examples"]).format(**context)
        prompt = PROMPTS["entity_extraction"].format(
            **context, examples=examples, input_text=document.text
        )
        messages = [{"role": "user", "content": prompt}]
        result = await self.model.text(
            messages, cache_namespace="knowledge_extraction", validator=validate_extraction
        )
        history = [*messages, {"role": "assistant", "content": result}]
        for index in range(self.config.entity_extract_max_gleaning):
            continuation = PROMPTS["entiti_continue_extraction"]
            request = [*history, {"role": "user", "content": continuation}]
            extra = await self.model.text(
                request,
                cache_namespace="knowledge_extraction_continuation",
                validator=lambda value: validate_extraction(value, prefix=result),
            )
            result += extra
            history = [*request, {"role": "assistant", "content": extra}]
            if index == self.config.entity_extract_max_gleaning - 1:
                break

            def decision(value: str) -> str:
                if value.strip().strip('"').strip("'").lower() not in ("yes", "no"):
                    raise IntegrityError("Invalid extraction continuation decision")
                return value

            check = await self.model.text(
                [
                    *history,
                    {"role": "user", "content": PROMPTS["entiti_if_loop_extraction"]},
                ],
                cache_namespace="knowledge_extraction_loop_check",
                validator=decision,
            )
            if check.strip().strip('"').strip("'").lower() != "yes":
                break
        return result

    async def query_entities(self, query: Query) -> list[str]:
        prompt = PROMPTS["query_entity_extraction"].format(
            query=query.semantic_text(),
            examples="\n\n".join(PROMPTS["query_entity_extraction_examples"]),
        )

        result = await self.model.text(
            [{"role": "user", "content": prompt}], cache_namespace="query_entity_extraction"
        )
        return native_operate._parse_query_entities(result)


async def extract_units(
    documents: Iterable[Document],
    extractor: Extractor,
    store: Storage,
    concurrency: int,
) -> None:
    """Extract documents with bounded workers and per-document checkpoints."""
    queue: asyncio.Queue[Document] = asyncio.Queue()
    for d in documents:
        queue.put_nowait(d)
    total = queue.qsize()
    completed = 0
    stop = asyncio.Event()
    errors = []

    async def worker() -> None:
        nonlocal completed
        while not queue.empty() and (not stop.is_set()):
            try:
                d = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            chunk = store.get("text_chunks", d.id) or {"content": d.text, "source": d.source}
            if chunk.get("extraction") is not None:
                continue
            logger.info("Extraction started chunk_id=%s", d.id)
            started = perf_counter()
            try:
                result = await extractor.facts(d)
                chunk["extraction"] = result
                store.put("text_chunks", d.id, chunk)
                completed += 1
                logger.info(
                    "Extraction checkpointed chunk_id=%s progress=%d/%d elapsed_seconds=%.3f",
                    d.id, completed, total, perf_counter() - started,
                )
            except Exception as error:
                logger.error("Extraction failed chunk_id=%s error=%s", d.id, type(error).__name__)
                errors.append(error)
                stop.set()
                return

    tasks = [
        asyncio.create_task(worker()) for _ in range(min(concurrency, queue.qsize()))
    ]
    try:
        await asyncio.gather(*tasks)
        if errors:
            raise errors[0]
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


# Knowledge hypergraph construction
class KnowledgeGraph:
    def __init__(self, store: Storage, model: CachedModel, config: BuildConfig) -> None:
        self.store, self.model, self.config = store, model, config
        self.graph = store.graph("knowledge")
        self.entities_vdb = store.vdb("entities")
        self.hyperedges_vdb = store.vdb("hyperedges")
        self.chunks_vdb = store.vdb("chunks")

    @property
    def count(self) -> int:
        return self.graph._graph.number_of_nodes()

    def node(self, key: str) -> dict | None:
        data = self.graph._graph.nodes.get(key)
        if data is None:
            return None
        row = dict(data)
        row.update(
            id=key,
            kind=row["role"],
            sources=sorted(set(row["source_id"].split(GRAPH_FIELD_SEP))),
        )
        if row["role"] == "entity":
            row.update(
                text=key + " [" + row["entity_type"] + "]: " + row["description"],
            )
        else:
            row.update(
                text=key.removeprefix("<hyperedge>"),
                members=sorted(self.graph._graph.neighbors(key)),
            )
        return row

    def vector(self, key: str) -> tuple[np.ndarray, dict] | None:
        row = self.node(key)
        if row is None:
            return None
        prefix = "ent-" if row["kind"] == "entity" else "rel-"
        return self.store.vector(
            "knowledge", native_operate.compute_mdhash_id(key, prefix=prefix)
        )

    def vectors(self, *, copy_vectors: bool = True) -> list[tuple[str, np.ndarray, dict]]:
        return [
            (meta["graph_id"], vector, meta)
            for _, vector, meta in self.store.vectors("knowledge", copy_vectors=copy_vectors)
        ]

    async def embed_chunks(self, documents: Iterable[Document]) -> None:
        """Embed raw text chunks in the knowledge source index."""
        missing = [
            doc
            for doc in documents
            if self.store.vector("knowledge_chunks", doc.id) is None
        ]
        if not missing:
            return
        await self.model.embed([doc.text for doc in missing])
        await self.store.upsert_vectors(
            "chunks", {doc.id: {"content": doc.text} for doc in missing}
        )

    async def ingest_batch(self, units: Iterable[tuple[Document, str]]) -> bool:
        units = [
            (doc, data)
            for doc, data in units
            if not (self.store.get("text_chunks", doc.id) or {}).get("indexed")
        ]
        if not units:
            logger.info("KHG merge reused all indexed chunks")
            return False
        logger.info("KHG merge started chunks=%d", len(units))
        native = native_operate
        merge_graph = copy(self.graph)
        merge_graph._graph = self.graph._graph.copy()
        entities, hyperedges = defaultdict(list), defaultdict(list)
        for document, extracted in units:
            now_hyper_relation = ""
            for record_attributes in extraction_records(extracted):
                relation = await native._handle_single_hyperrelation_extraction(
                    record_attributes, document.id
                )
                if relation is not None:
                    hyperedges[relation["hyper_relation"]].append(relation)
                    now_hyper_relation = relation["hyper_relation"]
                entity = await native._handle_single_entity_extraction(
                    record_attributes, document.id, now_hyper_relation
                )
                if entity is not None:
                    entities[entity["entity_name"]].append(entity)

        async def summarize(prompt: str, **parameters) -> str:
            target_tokens = parameters["max_tokens"] * 4 // 5
            return await self.model.text(
                [
                    {
                        "role": "system",
                        "content": ENTITY_SUMMARY_INSTRUCTION.format(
                            target_tokens=target_tokens,
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                cache_namespace="entity_description_summary",
                **parameters,
            )

        global_config = {
            "llm_model_func": summarize,
            "entity_summary_to_max_tokens": self.config.entity_summary_to_max_tokens,
            "llm_model_max_token_size": self.config.llm_model_max_token_size,
            "tiktoken_model_name": self.config.tokenizer_model,
            "addon_params": {},
        }
        for name, records in hyperedges.items():
            await native._merge_hyperedges_then_upsert(
                name, records, merge_graph, global_config
            )
        for name, records in entities.items():
            await native._merge_nodes_then_upsert(
                name, records, merge_graph, global_config
            )
        for name, records in entities.items():
            await native._merge_edges_then_upsert(
                name, records, merge_graph, global_config
            )

        changed = list(hyperedges) + list(entities)
        contents = []
        metadata_rows = {}
        for name in changed:
            row = dict(merge_graph._graph.nodes[name])
            row["id"] = name
            row["kind"] = row["role"]
            if row["role"] == "entity":
                row["embedding_content"] = name + row["description"]
                row["vector_id"] = native.compute_mdhash_id(name, prefix="ent-")
            else:
                row["embedding_content"] = name
                row["vector_id"] = native.compute_mdhash_id(name, prefix="rel-")
            metadata_rows[name] = row
            contents.append(row["embedding_content"])
        # Hyperedges encode their full names; entities encode name + description.
        if contents:
            await self.model.embed(contents)
        with self.store.transaction():
            # Embedding responses are durable before graph/index publication.
            for kind, space in (("hyperedge", "hyperedges"), ("entity", "entities")):
                data = {}
                for row in metadata_rows.values():
                    if row["kind"] != kind:
                        continue
                    data[row["vector_id"]] = {
                        "content": row["embedding_content"],
                        "kind": kind,
                        "graph_id": row["id"],
                        "score": row["score"],
                        "entity_name" if kind == "entity" else "hyperedge_name": row[
                            "id"
                        ],
                    }
                await self.store.upsert_vectors(space, data)
            self.store.set_graph("knowledge", merge_graph._graph)
            for document, _ in units:
                chunk = self.store.get("text_chunks", document.id) or {"content": document.text, "source": document.source}
                chunk["indexed"] = True
                self.store.put("text_chunks", document.id, chunk)
        return True

    async def soft_links(self, case_vector: np.ndarray) -> list[dict]:
        # No suspension or index mutation occurs while these read-only views live.
        rows = self.vectors(copy_vectors=False)
        if not rows:
            return []
        scores = np.asarray(
            [float(case_vector @ vector) for _, vector, _ in rows], dtype=np.float64
        )
        order = sorted(
            (
                i for i, score in enumerate(scores)
                if score > self.config.cosine_better_than_threshold
            ),
            key=lambda i: (-scores[i], rows[i][0]),
        )[: self.config.softlink_k]
        return [{"node_id": rows[i][0], "weight": float(scores[i])} for i in order]

    async def search(
        self, query_text: str, kind: str, k: int,
    ) -> list[RankedNode]:
        index = self.entities_vdb if kind == "entity" else self.hyperedges_vdb
        hits = await index.query(query_text, top_k=k)
        return [
            (
                cast(dict, self.node(hit["graph_id"])),
                float(hit["distance"]),
                cast(tuple[np.ndarray, dict], self.vector(hit["graph_id"]))[0],
            )
            for hit in hits
        ]


# Case identities are separate from medical attribute nodes.
def attribute_tuple(content: object) -> list[dict]:
    """Flatten medical fields into attribute paths and values."""
    result: list[dict] = []

    def walk(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, v in value.items():
                walk(v, path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, dict):
                    result.append({"attribute": path, "value": v})
                else:
                    walk(v, path)
        elif value is not None:
            result.append({"attribute": path, "value": value})

    walk(content, "")
    return result


async def build_case(
    case: Case,
    store: Storage,
    model: CachedModel,
    knowledge: KnowledgeGraph,
    knowledge_revision: str | None = None,
) -> None:
    """Index one pending reference case selected by the construction stage."""
    await build_case_batch([case], store, model, knowledge, knowledge_revision)


async def build_case_batch(
    cases: Sequence[Case], store: Storage, model: CachedModel,
    knowledge: KnowledgeGraph, knowledge_revision: str | None = None,
) -> None:
    """Prepare cases before publishing their graph, records and vectors together."""
    if not cases:
        return
    vectors = await model.embed([case["case_text"] for case in cases])
    prepared = []
    try:
        for case, vector in zip(cases, vectors):
            content = clinical_content(case)
            prepared.append({
                "case_id": case["case_id"],
                "subject_id": case["subject_id"],
                "domain": case["domain_id"],
                "content": content,
                "attributes": attribute_tuple(content),
                "text": case["case_text"],
                "soft_links": await knowledge.soft_links(vector),
                "knowledge_revision": knowledge_revision,
            })
    except BaseException:
        # Preserve the successfully prepared prefix when a later case fails.
        await _publish_cases(prepared, store, knowledge)
        raise
    await _publish_cases(prepared, store, knowledge)


async def _publish_cases(rows: list[dict], store: Storage, knowledge: KnowledgeGraph) -> None:
    if not rows:
        return
    with store.transaction():
        await store.upsert_vectors(
            "cases",
            {
                row["case_id"]: {
                    "content": row["text"],
                    "domain": row["domain"],
                    "subject_id": row["subject_id"],
                }
                for row in rows
            },
        )
        graph_name = "cases"
        native = store.graph(graph_name)
        graph = native._graph.copy()
        for row in rows:
            cid = row["case_id"]
            if cid in graph:
                previous_neighbors = list(graph.neighbors(cid))
                graph.remove_node(cid)
                graph.remove_nodes_from(
                    [node for node in previous_neighbors if graph.degree(node) == 0]
                )
            graph.add_node(cid, kind="case", subject_id=row["subject_id"])
            for attribute in row["attributes"]:
                aid = "attribute-" + identity(attribute)
                graph.add_node(aid, kind="attribute", attribute=canonical(attribute))
                graph.add_edge(cid, aid, kind="incidence")
            for link in row["soft_links"]:
                knowledge_node = cast(dict, knowledge.node(link["node_id"]))
                graph.add_node(
                    link["node_id"],
                    kind=knowledge_node["kind"],
                    text=knowledge_node["text"],
                )
                graph.add_edge(cid, link["node_id"], kind="soft", weight=link["weight"])
            store.put("cases", cid, row)
        store.set_graph(graph_name, graph)
    logger.info("CHG batch checkpointed cases=%d", len(rows))


# EEG representation and reference search
def paa(
    signal: ArrayLike, n: int = BuildConfig.paa_segments,
    z_normalize: bool = BuildConfig.z_normalize,
) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    # Eq.7 uses floor(T/n) and the first m*n samples, after optional normalization.
    m = x.shape[1] // n
    if z_normalize:
        std = x.std(axis=1, keepdims=True)
        x = (x - x.mean(axis=1, keepdims=True)) / std
    return x[:, :m * n].reshape(x.shape[0], n, m).mean(axis=2).astype(np.float32)


@njit
def _banded_dtw(
    a: np.ndarray, b: np.ndarray, radius: int, max_distance: float,
) -> float:
    n = a.shape[1]
    width = 2 * radius + 3
    prev = np.full(width, np.inf)
    curr = np.empty(width)
    costs = np.empty(2 * radius + 1)
    prev[radius + 1] = 0.0
    for i in range(1, n + 1):
        curr.fill(np.inf)
        start = max(1, i - radius)
        stop = min(n, i + radius)
        count = stop - start + 1
        costs[:count] = 0.0
        for c in range(a.shape[0]):
            for offset in range(count):
                costs[offset] += abs(a[c, i - 1] - b[c, start + offset - 1])
        row_min = np.inf
        for offset in range(count):
            k = start + offset - i + radius + 1
            value = costs[offset] + min(prev[k + 1], curr[k - 1], prev[k])
            curr[k] = value
            row_min = min(row_min, value)
        # Every complete path crosses this row; remaining costs are nonnegative.
        if row_min > max_distance:
            return np.inf
        prev, curr = curr, prev
    distance = prev[radius + 1]
    return distance if distance <= max_distance else np.inf


def eeg_distance(
    a: np.ndarray, b: np.ndarray, radius: int, max_distance: float = math.inf,
) -> float:
    """Shared-path L1-DTW; return infinity when distance exceeds the cutoff."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape:
        raise ValueError("EEG inputs must have the same (channel, time) shape")
    radius = min(radius, a.shape[1] - 1)
    if radius == 0:
        distance = float(np.abs(a - b).sum())
        return distance if distance <= max_distance else math.inf
    return float(_banded_dtw(a, b, radius, max_distance))


class EEGIndex:
    def __init__(
        self,
        store: Storage,
        corpus: DatasetCorpus,
        config: BuildConfig | None = None,
    ) -> None:
        self.store = store
        self.corpus = corpus
        self.config = config or BuildConfig()

    def build(self, domain: str) -> dict:
        logger.info("EVD build started domain=%s working_dir=%s", domain, self.store.directory)
        if domain not in self.corpus.dataset.domains:
            available = ", ".join(sorted(self.corpus.dataset.domains))
            raise ValueError(f"Unknown EEG domain {domain!r}. Available domains: {available}")
        revision = identity([self.corpus.signature, domain, asdict(self.config)])
        old_state = self.store.stage("eeg:" + domain)
        if (
            old_state
            and old_state["complete"]
            and (old_state.get("input_revision") == revision)
        ):
            logger.info("EVD build reused domain=%s windows=%s", domain, old_state.get("windows"))
            return {**old_state, "added": 0}
        self.store.set_stage(
            "eeg:" + domain, revision, False, input_revision=revision
        )
        added = total = 0
        try:
            for segment in self.corpus.reference_segments(domain):
                total += 1
                if total % 10000 == 0:
                    logger.info(
                        "EVD %s: inspected %s windows; newly encoded %s",
                        domain,
                        total,
                        added,
                    )
                try:
                    if self.insert(segment):
                        added += 1
                except Exception as error:
                    logger.error(
                        "EVD encoding failed domain=%s segment_id=%s error=%s",
                        domain, segment["segment_id"], type(error).__name__,
                    )
                    raise
        finally:
            self.store.eeg("eeg:" + domain).flush()
        logger.info(
            "EVD encoding finished domain=%s windows=%d added=%d reused=%d",
            domain, total, added, total - added,
        )
        return self.store.set_stage(
            "eeg:" + domain,
            identity(
                [revision, self.store.digest(vector_spaces=("eeg:" + domain,))]
            ),
            True,
            input_revision=revision,
            windows=total,
            completed_windows=total,
            added=added,
        )

    def insert(self, segment: Segment) -> bool:
        sid = segment["segment_id"]
        index = self.store.eeg("eeg:" + segment["domain_id"])
        if index.contains(sid):
            return False
        vector = paa(
            self.corpus.dataset.signal(sid),
            self.config.paa_segments,
            self.config.z_normalize,
        )
        index.put(
            sid,
            vector.reshape(-1),
            {
                "domain": segment["domain_id"],
                "subject_id": segment["subject_id"],
                "case_id": segment["case_id"],
                "shape": list(vector.shape),
                "record_id": segment["record_id"],
            },
        )
        return True

    def search(self, query: Query, k: int) -> list[Evidence]:
        if not query.eeg:
            return []
        cache_key = identity(
            [
                self.corpus.signature,
                query.domain,
                query.eeg,
                k,
                asdict(self.config),
                (self.store.get("stages", "eeg:" + query.domain) or {}).get("revision"),
            ]
        )
        cached = self.store.get("eeg_search", cache_key)
        if cached is not None:
            logger.info(
                "EVD search reused query_id=%s domain=%s neighbors=%d",
                query.id, query.domain, len(cached),
            )
            return [Evidence(**e) for e in cached]
        started = perf_counter()
        checked = 0
        logger.info(
            "EVD search started query_id=%s domain=%s query_windows=%d top_k=%d",
            query.id, query.domain, len(query.eeg), k,
        )
        nearest: list[tuple[float, str, dict]] = []
        queries: list[tuple[np.ndarray, int]] = []
        for sid in query.eeg:
            signal = self.corpus.dataset.signal(sid)
            segment = self.corpus.dataset.segment(sid)
            vector = paa(signal, self.config.paa_segments, self.config.z_normalize)
            block_samples = signal.shape[1] // self.config.paa_segments
            step_seconds = block_samples * segment["duration_seconds"] / signal.shape[1]
            radius = math.floor(self.config.dtw_window_seconds / step_seconds)
            queries.append((vector, radius))
        for sid, v, meta in self.store.vectors("eeg:" + query.domain):
            checked += 1
            if checked % 10000 == 0:
                logger.info(
                    "EVD search progress query_id=%s domain=%s scanned=%d elapsed_seconds=%.3f",
                    query.id, query.domain, checked, perf_counter() - started,
                )
            candidate = v.reshape(meta["shape"])
            worst = (
                max(range(len(nearest)), key=lambda i: (nearest[i][0], nearest[i][1]))
                if len(nearest) == k else None
            )
            cutoff = nearest[worst][0] if worst is not None else math.inf
            distance = min(
                eeg_distance(q, candidate, radius, cutoff)
                for q, radius in queries
            )
            entry = (distance, sid, meta)
            if worst is None:
                nearest.append(entry)
            elif (distance, sid) < nearest[worst][:2]:
                nearest[worst] = entry
        result = []
        for distance, sid, meta in sorted(nearest, key=lambda row: (row[0], row[1])):
            evidence = self.corpus.dataset.evidence(sid)
            result.append(
                Evidence(
                    sid,
                    "eeg",
                    evidence["text"],
                    -distance,
                    {
                        "case_id": meta["case_id"],
                        "subject_id": meta["subject_id"],
                        "segment_id": sid,
                        "event_ids": evidence["event_ids"],
                        "binding_level": evidence["binding_level"],
                        "distance": distance,
                    },
                )
            )
        self.store.put("eeg_search", cache_key, [asdict(e) for e in result])
        logger.info(
            "EVD search checkpointed query_id=%s domain=%s scanned=%d neighbors=%d "
            "elapsed_seconds=%.3f",
            query.id, query.domain, checked, len(result), perf_counter() - started,
        )
        return result


# Multilayer retrieval, neighborhood expansion and fusion
def mmr(
    candidates: Iterable[RankedNode], k: int, weight: float,
) -> list[RankedNode]:
    """Greedy maximal marginal relevance with deterministic tie-breaking."""
    remaining = list(candidates)
    chosen: list[RankedNode] = []
    while remaining and len(chosen) < k:
        ranked = []
        for i, (row, relevance, vector) in enumerate(remaining):
            redundant = max((float(vector @ v) for _, _, v in chosen), default=0.0)
            ranked.append(
                (-(weight * relevance - (1 - weight) * redundant), row["id"], i)
            )
        _, _, i = min(ranked)
        chosen.append(remaining.pop(i))
    return chosen


class Retriever:
    def __init__(
        self,
        corpus: DatasetCorpus | None,
        store: Storage,
        model: CachedModel,
        knowledge: KnowledgeGraph,
        eeg: EEGIndex | None,
        extractor: Extractor,
        tokenizer: Tokenizer,
        config: QueryConfig,
    ) -> None:
        self.corpus = corpus
        self.store = store
        self.model = model
        self.knowledge = knowledge
        self.eeg = eeg
        self.extractor = extractor
        self.tokenizer = tokenizer
        self.config = config

    async def retrieve_cases(self, query: Query) -> list[Evidence]:
        """Retrieve reference cases by narrative similarity and attribute incidence."""
        text_hits = await self.store.vdb("cases").query(
            query.semantic_text(), top_k=self.config.case_text_top_k,
        )
        text_ranks = {hit["id"]: rank for rank, hit in enumerate(text_hits, 1)}
        similarities = {hit["id"]: float(hit["distance"]) for hit in text_hits}
        facts = query_attributes(query.masked_case)
        matches: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
        if any(fact.primary for fact in facts):
            graph = self.store.graph("cases")._graph
            for node_id, node in graph.nodes(data=True):
                if node.get("kind") != "attribute":
                    continue
                attribute = json.loads(node["attribute"])
                hit_facts = [i for i, fact in enumerate(facts) if attribute_matches(fact, attribute)]
                if not hit_facts:
                    continue
                for cid in graph.neighbors(node_id):
                    if (
                        graph.nodes[cid].get("kind") == "case"
                        and graph.edges[node_id, cid].get("kind") == "incidence"
                    ):
                        for i in hit_facts:
                            matches[cid][i].add(node_id)
        matches = {
            cid: matched for cid, matched in matches.items()
            if any(facts[i].primary for i in matched)
        }
        if matches:
            vector = (await self.model.embed([query.semantic_text()]))[0]
            for cid, candidate, _ in self.store.vectors("cases"):
                if cid in matches:
                    similarities[cid] = float(vector @ candidate)
        attribute_ids = sorted(
            matches,
            key=lambda cid: (-len(matches[cid]), -similarities[cid], cid),
        )[:self.config.case_attribute_top_k]
        attribute_ranks = {cid: rank for rank, cid in enumerate(attribute_ids, 1)}
        scores = {
            cid: sum(
                1.0 / (self.config.case_rrf_k + ranks[cid])
                for ranks in (text_ranks, attribute_ranks) if cid in ranks
            )
            for cid in text_ranks.keys() | attribute_ranks.keys()
        }
        selected = sorted(
            scores,
            key=lambda cid: (
                -scores[cid], text_ranks.get(cid, math.inf),
                attribute_ranks.get(cid, math.inf), cid,
            ),
        )[:self.config.case_top_k]
        result = []
        for cid in selected:
            row = self.store.get("cases", cid)
            matched = matches.get(cid, {})
            result.append(Evidence(
                cid, "case", row["text"], scores[cid],
                {
                    "case_id": cid, "subject_id": row["subject_id"],
                    "case_retrieval": {
                        "text_rank": text_ranks.get(cid),
                        "attribute_rank": attribute_ranks.get(cid),
                        "matched_attributes": [
                            {**asdict(facts[i]), "node_ids": sorted(nodes)}
                            for i, nodes in sorted(matched.items())
                        ],
                    },
                },
                {"case_rrf": scores[cid], "case_cosine": similarities[cid],
                 "attribute_matches": len(matched)},
            ))
        logger.info(
            "CHG retrieval query_id=%s text_candidates=%d attribute_candidates=%d cases=%d",
            query.id, len(text_hits), len(matches), len(result),
        )
        return result

    async def retrieve_candidates(self, query: Query) -> RetrievalCandidates:
        """Retrieve, expand and rank each evidence category independently."""
        graph = nx.Graph()
        case_rows = {}
        case_hits = []
        anchors = []
        if query.case_id is not None:
            case_hits = [
                replace(item, provenance=dict(item.provenance), components=dict(item.components))
                for item in await self.retrieve_cases(query)
            ]
            for item in case_hits:
                case_rows[item.id] = self.store.get("cases", item.id)
                graph.add_node(item.id)

        eeg_matches: dict[str, list[dict]] = defaultdict(list)
        if query.case_id is not None and query.eeg:
            anchors = cast(EEGIndex, self.eeg).search(query, self.config.top_k)
            for item in anchors:
                cid = item.provenance["case_id"]
                eeg_matches[cid].append({
                    "segment_id": item.id, "distance": item.provenance["distance"],
                })
                graph.add_edge(item.id, cid)
                if cid not in case_rows:
                    case_rows[cid] = self.store.get("cases", cid)

        best_distances = {
            cid: min(item["distance"] for item in matches)
            for cid, matches in eeg_matches.items()
        }
        case_order = [item.id for item in case_hits]
        associated = sorted(
            best_distances.keys() - set(case_order),
            key=lambda cid: (best_distances[cid], cid),
        )
        case_order.extend(associated)
        for item in case_hits:
            if item.id in eeg_matches:
                item.provenance["eeg_matches"] = eeg_matches[item.id]
                item.components["eeg_distance"] = best_distances[item.id]
        for cid in associated:
            row = case_rows[cid]
            case_hits.append(Evidence(
                cid, "case", row["text"], -best_distances[cid],
                {"case_id": cid, "subject_id": row["subject_id"],
                 "eeg_matches": eeg_matches[cid]},
                {"eeg_distance": best_distances[cid]},
            ))

        case_texts = {cid: case_rows[cid]["text"] for cid in case_order}

        entities = []
        hyperedges = []
        chunks = []
        vector = (await self.model.embed([query.semantic_text()]))[0]
        hypers = mmr(
            await self.knowledge.search(
                query.semantic_text(), "hyperedge", self.config.candidate_pool,
            ),
            self.config.hyperedge_top_k,
            self.config.mmr_lambda,
        )
        entity_hits = []
        entity_query = ", ".join(await self.extractor.query_entities(query))
        if entity_query:
            entity_hits = await self.knowledge.search(
                entity_query, "entity", self.config.entity_top_k,
            )
        direct = {
            row["id"]: (rank, score)
            for hits in (entity_hits, hypers)
            for rank, (row, score, _) in enumerate(hits, 1)
        }
        links: dict[str, list[dict]] = defaultdict(list)
        for cid, row in case_rows.items():
            for link in row["soft_links"]:
                links[link["node_id"]].append({"case_id": cid, "weight": link["weight"]})
        seeds = set(direct) | set(links)
        kg = self.knowledge.graph._graph
        selected_nodes = set()
        expanded_from: dict[str, list[str]] = defaultdict(list)
        for seed in sorted(seeds):
            if seed in kg:
                neighborhood = nx.single_source_shortest_path_length(
                    kg, seed, cutoff=self.config.expansion_radius,
                )
                selected_nodes.update(neighborhood)
                for node, distance in neighborhood.items():
                    if distance:
                        expanded_from[node].append(seed)
        graph = nx.compose(graph, kg.subgraph(selected_nodes).copy())
        chunk_nodes: dict[str, set[str]] = defaultdict(set)
        for node in sorted(selected_nodes):
            row = cast(dict, self.knowledge.node(node))
            stored = cast(tuple[np.ndarray, dict], self.knowledge.vector(node))
            cosine = float(vector @ stored[0])
            origin = {}
            if node in direct:
                origin["direct_rank"] = direct[node][0]
                for sid in row["sources"]:
                    chunk_nodes[sid].add(node)
            if node in links:
                origin["soft_links"] = links[node]
            if node in expanded_from:
                origin["expanded_from"] = expanded_from[node]
            sources = [
                {"source_id": sid, **self.store.get("text_chunks", sid)["source"]}
                for sid in row["sources"]
            ]
            item = Evidence(
                node, "knowledge", row["text"],
                direct[node][1] if node in direct else cosine,
                {"sources": sources, "node_kind": row["kind"], "retrieval": origin},
                {"query_cosine": cosine},
            )
            (entities if row["kind"] == "entity" else hyperedges).append(item)
        for items in (entities, hyperedges):
            items.sort(key=lambda item: (
                0 if item.id in direct else 1,
                direct[item.id][0] if item.id in direct else -item.components["query_cosine"],
                item.id,
            ))
        for sid, nodes in chunk_nodes.items():
            chunk = self.store.get("text_chunks", sid)
            stored = cast(tuple[np.ndarray, dict], self.store.vector("knowledge_chunks", sid))
            cosine = float(vector @ stored[0])
            chunks.append(Evidence(
                sid, "source", chunk["content"], cosine,
                {"sources": [{"source_id": sid, **chunk["source"]}],
                 "retrieval_nodes": sorted(nodes)},
                {"query_cosine": cosine},
            ))
        chunks.sort(key=lambda item: (-item.score, item.id))
        for cid, row in case_rows.items():
            for link in row["soft_links"]:
                if link["node_id"] in selected_nodes:
                    graph.add_edge(cid, link["node_id"], weight=link["weight"])

        trace = {
            "nodes": sorted(graph.nodes),
            "edges": sorted(sorted(pair) for pair in graph.edges),
            "config": asdict(self.config),
            "case_retrieval": [
                {"case_id": item.id, **item.components, **item.provenance["case_retrieval"]}
                for item in case_hits if "case_retrieval" in item.provenance
            ],
            "candidate_order": {
                "entities": [item.id for item in entities],
                "hyperedges": [item.id for item in hyperedges],
                "cases": [item.id for item in case_hits],
                "eeg": [item.id for item in anchors],
                "chunks": [item.id for item in chunks],
            },
        }
        logger.info(
            "Retrieval candidates query_id=%s entities=%d hyperedges=%d cases=%d eeg=%d chunks=%d",
            query.id, len(entities), len(hyperedges), len(case_hits), len(anchors), len(chunks),
        )
        return RetrievalCandidates(
            entities=entities, hyperedges=hyperedges, cases=case_hits, eeg=anchors, chunks=chunks,
            case_texts=case_texts, case_order=case_order, trace=trace,
        )

    async def retrieve(self, query: Query) -> tuple[list[Evidence], dict]:
        candidates = await self.retrieve_candidates(query)
        selected, context_trace = ContextBuilder(candidates, self.tokenizer, self.config).build()
        trace = {**candidates.trace, **context_trace}
        context_tokens = trace["context_tokens"]
        candidate_count = sum(len(items) for items in (
            candidates.entities, candidates.hyperedges, candidates.chunks,
            candidates.cases, candidates.eeg,
        ))
        logger.info(
            "Retrieval evidence query_id=%s candidates=%d selected=%d context_tokens=%d budget=%d",
            query.id, candidate_count, len(selected), context_tokens,
            self.config.context_tokens,
        )
        logger.info("Context partitions query_id=%s budget=%s", query.id, trace["context_budget"])
        return selected, trace
