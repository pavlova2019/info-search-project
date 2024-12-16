import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex
)

from transformers import PreTrainedModel

import src.vector_store.store_config as store_config
import src.config as cfg
from src.util.article import Article
from src.parsing.sample_parser import load_articles


def collect_and_write_index(embed_model: PreTrainedModel,
                            index_path: str = cfg.VECTOR_INDEX,
                            articles_path: str = cfg.TEST_DATASET):
    articles: List[Article] = load_articles(articles_path)
    
    transformations = store_config.get_transformations()
    vector_store = FaissVectorStore(faiss_index=store_config.get_faiss_index())
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex.from_documents(
        documents=articles,
        storage_context=storage_context,
        transformations=transformations,
        embed_model=embed_model
    )
    
    index.storage_context.persist(persist_dir=index_path)
    
    return index
    

def load_index(embed_model: PreTrainedModel,
               index_path: str = cfg.VECTOR_INDEX):
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
