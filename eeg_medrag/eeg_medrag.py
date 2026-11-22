"""EEG-MedRAG framework: independent build stages and guided querying."""

from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
from typing import cast
from pathlib import Path

from dataclasses import asdict
import logging

from .base import BuildConfig, Document, Evidence, LLMConfig, ModelProvider, QueryConfig, Query, RetrievalCandidates
from .dataset import DatasetCorpus
from .generation import AnswerGenerator, join_evidence_text
from .llm import LLMProvider
from .operate import (
    EEGIndex,
    Extractor,
    KnowledgeGraph,
    Retriever,
    build_case_batch,
    extract_units,
)
from .prompt import CONTEXT_SECTIONS, PROMPTS
from .storage import CachedModel, Storage
from .utils import (
    HyperTokenizer,
    IncompleteStage,
    IntegrityError,
    PDFCorpus,
    component_descriptor,
    identity,
)


logger = logging.getLogger(__name__)


class EEGMedRAG:
    def __init__(
        self,
        working_dir: str | Path,
        corpus: DatasetCorpus | None = None,
        provider: ModelProvider | None = None,
        config: BuildConfig | None = None,
        tokenizer: HyperTokenizer | None = None,
        *,
        extractor_factory: Callable[..., Extractor] = Extractor,
        knowledge_factory: Callable[..., KnowledgeGraph] = KnowledgeGraph,
        eeg_index_factory: Callable[..., EEGIndex] = EEGIndex,
        retriever_factory: Callable[..., Retriever] = Retriever,
        generator_factory: Callable[..., AnswerGenerator] = AnswerGenerator,
        answer_provider: ModelProvider | None = None,
        storage_factory: Callable[..., Storage] = Storage,
    ) -> None:
        provider = provider or LLMProvider()
        self.corpus = corpus
        self.config = config or BuildConfig()
        self.store = storage_factory(working_dir)
        self.tokenizer = tokenizer or HyperTokenizer(self.config.tokenizer_model)
        self.model = CachedModel(provider, self.store)
        answer_provider = answer_provider or provider
        self.answer_model = (
            self.model if answer_provider is provider
            else CachedModel(answer_provider, self.store)
        )
        dimension = getattr(provider, "embedding_dimension", None) or getattr(
            getattr(provider, "config", None), "embedding_dimension", None
        )
        self.store.bind_model(
            self.model, dimension,
            getattr(getattr(provider, "config", None), "embedding_batch_size", LLMConfig.embedding_batch_size),
            self.config.cosine_better_than_threshold,
        )
        self.full_docs = self.store.kv("full_docs")
        self.text_chunks = self.store.kv("text_chunks")
        self.llm_response_cache = self.store.kv("llm")
        self.extractor = extractor_factory(self.model, self.config)
        self.knowledge = knowledge_factory(self.store, self.model, self.config)
        self.chunk_entity_relation_graph = self.knowledge.graph
        self.entities_vdb = self.knowledge.entities_vdb
        self.hyperedges_vdb = self.knowledge.hyperedges_vdb
        self.chunks_vdb = self.knowledge.chunks_vdb
        self.cases_vdb = self.store.vdb("cases")
        self.eeg = eeg_index_factory(self.store, corpus, self.config) if corpus is not None else None
        self.retriever_factory = retriever_factory
        self.generator = generator_factory(self.answer_model, self.tokenizer)
        self.retrieval_signature = identity([
            component_descriptor(retriever_factory), CONTEXT_SECTIONS, PROMPTS["query_entity_extraction"],
            PROMPTS["query_entity_extraction_examples"],
        ])
        self.knowledge_construction_signature = identity([asdict(self.config), self.model.signature])
        self.construction_signature = identity([
            self.knowledge_construction_signature, corpus.signature if corpus is not None else None,
        ])

    def stage(self, name: str) -> dict | None:
        return self.store.stage(name)

    def _require(self, name: str) -> dict:
        state = self.stage(name)
        if not state or not state["complete"]:
            raise IncompleteStage("Build stage is missing/incomplete: " + name)
        return state

    def pdf_documents(self, contexts_root: str | Path) -> list[Document]:
        return PDFCorpus(self.store, self.tokenizer, self.config).load(
            contexts_root
        )

    async def build_knowledge(self, documents: Iterable[Document]) -> dict:
        documents = sorted(documents, key=lambda d: d.id)
        logger.info(
            "KHG build started supplied_chunks=%d working_dir=%s",
            len(documents), self.store.directory,
        )
        with self.store.transaction():
            for doc in documents:
                row = self.store.get("text_chunks", doc.id) or {}
                source = doc.source
                if row.get("source") and row["source"] != source:
                    origins = row["source"].get("origins", [row["source"]])
                    origins += [origin for origin in source.get("origins", [source]) if origin not in origins]
                    source = {"origins": origins}
                row.update(content=doc.text, source=source, registered_source=identity(source))
                self.store.put("text_chunks", doc.id, row)
            chunks = [(key, row) for key, row in self.store.items("text_chunks") if "registered_source" in row]
            ids = [key for key, _ in chunks]
            revision = identity([
                self.knowledge_construction_signature,
                [(key, row["content"], row["source"]) for key, row in chunks],
            ])

            old = self.stage("knowledge")
            if old and old["complete"] and (old.get("input_revision") == revision):
                logger.info("KHG build reused chunks=%d revision=%s", len(ids), old["revision"][:12])
                return old
            self.store.set_stage("knowledge", revision, False, input_revision=revision)
        todo = []
        for did in ids:
            row = self.store.get("text_chunks", did)
            if row.get("extraction") is not None:
                continue
            todo.append(Document(did, row["content"], row["source"]))
        logger.info(
            "KHG extraction plan chunks=%d cached=%d pending=%d concurrency=%d",
            len(ids), len(ids) - len(todo), len(todo), self.config.concurrency,
        )
        await self.knowledge.embed_chunks(todo)
        await extract_units(
            todo, self.extractor, self.store, self.config.concurrency
        )
        # Merge after extraction completes; per-document checkpoints persist
        # independently of graph and vector publication.
        units = []
        for did in ids:
            row = self.store.get("text_chunks", did)
            units.append(
                (
                    Document(did, row["content"], row["source"]),
                    row["extraction"],
                )
            )
        await self.knowledge.ingest_batch(units)
        if not self.knowledge.count:
            raise IntegrityError(
                "No clinical facts extracted; inspect source/extraction outputs"
            )
        logger.info("KHG indexing finished chunks=%d nodes=%d", len(ids), self.knowledge.count)
        return self.store.set_stage(
            "knowledge",
            identity(
                [
                    revision,
                    self.store.digest(
                        graph_spaces=("knowledge",),
                        vector_spaces=("knowledge", "knowledge_chunks"),
                    ),
                ]
            ),
            True,
            input_revision=revision,
            units=len(ids),
            completed_units=sum(bool(row.get("indexed")) for _, row in self.store.items("text_chunks")),
            completed_extraction_units=sum(row.get("extraction") is not None for _, row in self.store.items("text_chunks")),
        )

    async def build_cases(self) -> dict:
        """Build one shared CHG from reference patients across all disease domains."""
        logger.info("CHG build started working_dir=%s", self.store.directory)
        kstate = self.stage("knowledge")
        if kstate and not kstate["complete"]:
            raise IncompleteStage("Finish KHG before linking CHG")
        kg_revision = kstate["revision"] if kstate else None
        revision = identity([self.construction_signature, kg_revision])
        old_state = self.stage("cases")
        if (
            old_state
            and old_state["complete"]
            and (old_state.get("input_revision") == revision)
        ):
            logger.info(
                "CHG build reused cases=%s revision=%s",
                old_state.get("cases"), old_state["revision"][:12],
            )
            return {**old_state, "added": 0}
        rows = list(cast(DatasetCorpus, self.corpus).reference_cases())
        self.store.set_stage(
            "cases",
            revision,
            False,
            knowledge_revision=kg_revision,
            input_revision=revision,
        )
        pending = [
            case
            for case in rows
            if (self.store.get("cases", case["case_id"]) or {}).get(
                "knowledge_revision", "missing"
            )
            != kg_revision
        ]
        logger.info(
            "CHG build plan reference_cases=%d existing=%d pending=%d",
            len(rows), len(rows) - len(pending), len(pending),
        )
        if pending:
            await self.model.embed(
                [case["case_text"] for case in pending]
            )
        for start in range(0, len(pending), self.store.case_batch_size):
            batch = pending[start : start + self.store.case_batch_size]
            logger.info(
                "CHG batch started cases=%d completed=%d/%d",
                len(batch), start, len(pending),
            )
            await build_case_batch(
                batch,
                self.store,
                self.model,
                self.knowledge,
                kg_revision,
            )
        logger.info("CHG indexing finished reference_cases=%d added=%d", len(rows), len(pending))
        return self.store.set_stage(
            "cases",
            identity(
                [
                    revision,
                    self.store.digest(
                        kv_spaces=("cases",),
                        vector_spaces=("cases",),
                        keys=[case["case_id"] for case in rows],
                    ),
                ]
            ),
            True,
            input_revision=revision,
            knowledge_revision=kg_revision,
            added=len(pending),
            cases=len(rows),
        )

    def build_eeg(self, domain: str) -> dict:
        return cast(EEGIndex, self.eeg).build(domain)

    def _require_stages(self, query: Query) -> None:
        """Stop queries that depend on unfinished construction outputs."""
        knowledge = self._require("knowledge")
        if query.case_id is not None:
            state = self._require("cases")
            expected = identity([self.construction_signature, state["knowledge_revision"]])
            if state.get("input_revision") != expected:
                raise IncompleteStage("Reference inputs changed; rebuild cases")
            if state["knowledge_revision"] != knowledge["revision"]:
                raise IncompleteStage("KHG changed; rebuild dependent case soft links")
        if query.case_id is not None and query.eeg:
            state = self._require("eeg:" + query.domain)
            expected = identity([cast(DatasetCorpus, self.corpus).signature, query.domain, asdict(self.config)])
            if state.get("input_revision") != expected:
                raise IncompleteStage("Reference inputs changed; rebuild EEG for " + query.domain)

    async def retrieve_candidates(
        self,
        query: Query,
        config: QueryConfig | None = None,
    ) -> RetrievalCandidates:
        """Return complete ranked candidates before context selection."""
        config = config or QueryConfig()
        self._require_stages(query)
        retriever = self.retriever_factory(
            self.corpus, self.store, self.model, self.knowledge, self.eeg,
            self.extractor, self.tokenizer, config,
        )
        return await retriever.retrieve_candidates(query)

    async def retrieve(
        self,
        query: Query,
        config: QueryConfig | None = None,
    ) -> tuple[list[Evidence], dict]:
        config = config or QueryConfig()
        self._require_stages(query)
        retriever = self.retriever_factory(
            self.corpus, self.store, self.model, self.knowledge, self.eeg,
            self.extractor, self.tokenizer, config,
        )
        return await retriever.retrieve(query)

    async def answer(self, query: Query, config: QueryConfig | None = None) -> dict:
        config = config or QueryConfig()
        evidence, trace = await self.retrieve(query, config)
        return await self.generate(query, evidence, config, trace=trace)

    async def generate(
        self,
        query: Query,
        evidence: Sequence[Evidence],
        config: QueryConfig | None = None,
        *,
        trace: dict | None = None,
    ) -> dict:
        """Generate an answer from supplied retrieval evidence."""
        config = config or QueryConfig()
        value = await self.generator.generate(query, join_evidence_text(evidence), config)
        return {
            "id": query.id, **value,
            "retrieval": {"evidence": [asdict(item) for item in evidence], "trace": trace or {}},
        }

    def status(self) -> dict[str, dict]:
        return {key: value for key, value in self.store.items("stages")}

    def artifact_revisions(
        self,
        queries: Iterable[Query],
    ) -> dict[str, str | None]:
        """Return the build revisions used by these queries."""
        names = set()
        for query in queries:
            names.add("knowledge")
            if query.case_id is not None:
                names.add("cases")
                if query.eeg:
                    names.add("eeg:" + query.domain)
        return {name: (self.stage(name) or {}).get("revision") for name in sorted(names)}


