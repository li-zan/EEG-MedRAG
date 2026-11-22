import asyncio
import os
from tqdm.asyncio import tqdm as tqdm_async
from dataclasses import dataclass
from typing import Union
import networkx as nx
import numpy as np
from nano_vectordb import NanoVectorDB

from .utils import (
    logger,
    load_json,
    write_json,
    # compute_mdhash_id,
)

from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if id in self._data
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}


@dataclass
class NanoVectorDBStorage(BaseVectorStorage):
    cosine_better_than_threshold: float = 0.2

    def __post_init__(self):
        self._client_file_name = os.path.join(
            self.global_config["working_dir"], f"vdb_{self.namespace}.json"
        )
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._client = NanoVectorDB(
            self.embedding_func.embedding_dim, storage_file=self._client_file_name
        )
        self.cosine_better_than_threshold = self.global_config.get(
            "cosine_better_than_threshold", self.cosine_better_than_threshold
        )

    def _validate_embedding_dimension(self, embeddings):
        shape = np.shape(embeddings)
        expected_dim = self.embedding_func.embedding_dim
        if len(shape) != 2 or shape[1] != expected_dim:
            raise ValueError(
                f"Embedding shape {shape} does not match embedding_dim={expected_dim}; "
                "use an EmbeddingFunc with matching dimensions."
            )

    async def upsert(self, data: dict[str, dict]):
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You insert an empty data to vector DB")
            return []
        list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        async def wrapped_task(batch):
            result = await self.embedding_func(batch)
            self._validate_embedding_dimension(result)
            pbar.update(1)
            return result

        embedding_tasks = [wrapped_task(batch) for batch in batches]
        pbar = tqdm_async(
            total=len(embedding_tasks), desc="Generating embeddings", unit="batch"
        )
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        if len(embeddings) == len(list_data):
            for i, d in enumerate(list_data):
                d["__vector__"] = embeddings[i]
            results = self._client.upsert(datas=list_data)
            return results
        else:
            # sometimes the embedding is not returned correctly. just log it.
            logger.error(
                f"embedding is not 1-1 with data, {len(embeddings)} != {len(list_data)}"
            )

    async def query(
        self, query: str, top_k=5, *,
        score_threshold: float = None,
    ):
        if top_k <= 0:
            return []
        embedding = await self.embedding_func([query])
        self._validate_embedding_dimension(embedding)
        embedding = embedding[0]
        threshold = self.cosine_better_than_threshold
        candidate_top_k = top_k
        if score_threshold is not None:
            # A cosine top-k shortlist can omit the best confidence-weighted hit.
            candidate_top_k = len(self._client)
            threshold = None
        results = self._client.query(
            query=embedding,
            top_k=candidate_top_k,
            better_than_threshold=threshold,
        )
        results = [
            {**dp, "id": dp["__id__"], "distance": dp["__metrics__"]} for dp in results
        ]
        if score_threshold is not None:
            for result in results:
                if "score" not in result:
                    raise RuntimeError(
                        "Index has no extraction confidence scores; "
                        "rebuild the corpus in a new working_dir."
                    )
                result["rank"] = result["distance"] * result["score"]
            results = sorted(
                (result for result in results if result["rank"] > score_threshold),
                key=lambda result: result["rank"], reverse=True,
            )[:top_k]
        return results

    # Disabled legacy LightRAG vector deletion
    # @property
    # def client_storage(self):
    #     return getattr(self._client, "_NanoVectorDB__storage")

    # async def delete_entity(self, entity_name: str):
    #     try:
    #         entity_id = [compute_mdhash_id(entity_name, prefix="ent-")]

    #         if self._client.get(entity_id):
    #             self._client.delete(entity_id)
    #             logger.info(f"Entity {entity_name} have been deleted.")
    #         else:
    #             logger.info(f"No entity found with name {entity_name}.")
    #     except Exception as e:
    #         logger.error(f"Error while deleting entity {entity_name}: {e}")

    # async def delete_relation(self, entity_name: str):
    #     try:
    #         relations = [
    #             dp
    #             for dp in self.client_storage["data"]
    #             if dp["src_id"] == entity_name or dp["tgt_id"] == entity_name
    #         ]
    #         ids_to_delete = [relation["__id__"] for relation in relations]

    #         if ids_to_delete:
    #             self._client.delete(ids_to_delete)
    #             logger.info(
    #                 f"All relations related to entity {entity_name} have been deleted."
    #             )
    #         else:
    #             logger.info(f"No relations found for entity {entity_name}.")
    #     except Exception as e:
    #         logger.error(
    #             f"Error while deleting relations for entity {entity_name}: {e}"
    #         )

    async def index_done_callback(self):
        self._client.save()


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"], f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            logger.info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()

    async def index_done_callback(self):
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    # Disabled legacy LightRAG node deletion
    # async def delete_node(self, node_id: str):
    #     """
    #     Delete a node from the graph based on the specified node_id.

    #     :param node_id: The node_id to delete
    #     """
    #     if self._graph.has_node(node_id):
    #         self._graph.remove_node(node_id)
    #         logger.info(f"Node {node_id} deleted from the graph.")
    #     else:
    #         logger.warning(f"Node {node_id} not found in the graph for deletion.")
