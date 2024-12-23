import os, time
import Stemmer
from typing import List, Dict, Optional, Set
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    QueryBundle
)
from llama_index.core.schema import NodeWithScore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import BaseDocumentStore
from transformers import PreTrainedModel

import src.storing.store_config as store_config
from src.util.article import Article
from src.parsing.sample_parser import load_articles
from src.db.db import save_logs


class CompositeRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        logs_path: str,
        mode: str = 'OR',
    ) -> None:
        if mode not in ('AND', 'OR'):
            raise ValueError('Invalid mode.')
        self._mode = mode
        
        self._retrievers = retrievers
        super().__init__()
        self.logs_path = logs_path

    # @staticmethod
    def _save_metrics(self, time_metric: Dict[int, float]):
        for num, t in time_metric.items():
            save_logs("retreiver_" + str(num), t, self.logs_path)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        results_by_retriever: List[List[NodeWithScore]] = []
        for i, retriever in enumerate(self._retrievers):
            start_time = time.time()
            retrieve_result = retriever.retrieve(query_bundle)
            self._save_metrics({i: time.time() - start_time})
            results_by_retriever.append(retrieve_result)

        if len(self._retrievers) > 1:
            combined_dict: Dict[str, NodeWithScore] = {
                node.node.node_id: node
                for nodes in results_by_retriever
                for node in nodes
            }
    
            all_ids_by_retriever: List[Set[str]] = [
                {node.node.node_id for node in nodes}
                for nodes in results_by_retriever
            ]
    
            if self._mode == 'AND':
                retrieve_ids = set.intersection(*all_ids_by_retriever)
            else:  # "OR"
                retrieve_ids = set.union(*all_ids_by_retriever)

            output = [combined_dict[rid] for rid in retrieve_ids]

        else:
            output = results_by_retriever[0]

        return output
        

def collect_and_write_vector_index(embed_model: PreTrainedModel,
                                   index_path: str,
                                   articles_path: str):
    articles: List[Article] = load_articles(articles_path)
    embedding_dim = embed_model._model.get_sentence_embedding_dimension()
    
    transformations = store_config.get_transformations()
    vector_store = FaissVectorStore(faiss_index=store_config.get_faiss_index(embedding_dim))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=articles,
        storage_context=storage_context,
        transformations=transformations,
        embed_model=embed_model
    )
    
    index.storage_context.persist(persist_dir=index_path)
    
    return index


def collect_and_write_bm25_index(docstore: BaseDocumentStore,
                                 similarity_top_k: int,
                                 index_path: str):
    retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    retriever.persist(index_path)
    return retriever
    

def load_vector_index(embed_model: PreTrainedModel,
                      index_path: str):
    transformations = store_config.get_transformations()
    vector_store = FaissVectorStore.from_persist_dir(index_path)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=index_path
    )
    
    index = load_index_from_storage(
        storage_context=storage_context,
        transformations=transformations,
        embed_model=embed_model,
    )

    return index


def load_bm25_retriever(index_path: str):
    return BM25Retriever.from_persist_dir(index_path)


def load_retriever(vector_index_path: str,
                   embed_model: PreTrainedModel,
                   logs_path: str,
                   k_vector_search: int = 5,
                   article_path: Optional[str] = None,
                   bm25_index_path: Optional[str] = None,
                   k_text_search: int = 2):
    retrievers = []
    
    # check vector index 
    if os.path.exists(vector_index_path):
        vector_index = load_vector_index(embed_model, vector_index_path)
    else:
        if not article_path:
            raise ValueError(f"article_path")
        vector_index = collect_and_write_vector_index(embed_model, vector_index_path, article_path)

    vector_retriever = vector_index.as_retriever(similarity_top_k=k_vector_search)
    retrievers.append(vector_retriever)

    # check text index 
    if bm25_index_path:
        if os.path.exists(bm25_index_path):
            bm25_retriever = load_bm25_retriever(bm25_index_path)
            if bm25_retriever.similarity_top_k != k_text_search:
                bm25_retriever = collect_and_write_bm25_index(vector_index.docstore,
                                                              k_text_search,
                                                              bm25_index_path)
        else:
            bm25_retriever = collect_and_write_bm25_index(vector_index.docstore,
                                                          k_text_search,
                                                          bm25_index_path)
        retrievers.append(bm25_retriever)
    
    retriever = CompositeRetriever(retrievers, logs_path=logs_path)
    return retriever
