import os
import src.config as cfg
from typing import Optional
from src.db.db import setup_database
from src.text_gen.llm import load_llm
from src.embedders.embedder import load_embedder
from src.vector_store.vector_store import collect_and_write_index, load_index


def creating_query_engine(
    llm_model_name: str = cfg.llm_model_name,
    max_new_tokens: int = cfg.max_new_tokens,
    top_k_chunks: int = cfg.similarity_top_k,
    query_embed_model_name: str = cfg.embed_model_name,
    query_embed_kwargs: Optional[dict] = cfg.query_embed_kwargs,
    chunk_embed_model_name: str = cfg.embed_model_name,
    chunk_embed_kwargs: Optional[dict] = cfg.chunk_embed_kwargs,
    index_path: str = cfg.VECTOR_INDEX,
    rating_db_path: str = cfg.RATINGS_DB_PATH,
    article_path: str = cfg.TEST_DATASET,
):
    query_embed_model = load_embedder(query_embed_model_name, model_kwargs=query_embed_kwargs)
    chunk_embed_model = load_embedder(chunk_embed_model_name, model_kwargs=chunk_embed_kwargs)

    # check index 
    if os.path.exists(index_path):
        index = load_index(chunk_embed_model, index_path)
    else:
        index = collect_and_write_index(chunk_embed_model, index_path, article_path)

    # check ratings db
    if not os.path.exists(rating_db_path):
        setup_database()

    llm = load_llm(llm_model_name, max_new_tokens)

    query_engine = index.as_query_engine(
        embed_model=query_embed_model,
        llm=llm,
        similarity_top_k=top_k_chunks,
    )

    return query_engine


query_engine = creating_query_engine()

def query_rag_system(query):
    return str(query_engine.query(query))
