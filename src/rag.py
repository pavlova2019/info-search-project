import os
import yaml
import src.config
from typing import Optional, Dict
from src.db.db import setup_database
from src.text_gen.llm import load_llm_and_qa_tmpl
from src.embedders.embedder import load_embedder
from src.storing.storing import load_retriever
from llama_index.core.query_engine import RetrieverQueryEngine

# loading hyperparameters
with open('config.yaml', 'r') as f:
    hyps = yaml.safe_load(f)
    

def creating_query_engine(
    llm_model_name: str = hyps['llm_model']['model_name'],
    max_new_tokens: int = hyps['llm_model']['max_new_tokens'],
    query_embed_model_name: str = hyps['embed_model']['query_model_name'],
    query_embed_kwargs: Optional[Dict] = hyps['embed_model']['query_kwargs'],
    chunk_embed_model_name: str = hyps['embed_model']['chunk_model_name'],
    chunk_embed_kwargs: Optional[Dict] = hyps['embed_model']['chunk_kwargs'],
    vector_index_path: str = hyps['paths']['vector_index'],
    bm25_index_path: str = hyps['paths']['text_index'],
    k_vector_search: int = hyps['retriever']['k_vector_search'],
    k_text_search: int = hyps['retriever']['k_text_search'],
    article_path: str = hyps['paths']['dataset_w_articles'],
    cache_dir: str = hyps['paths']['cache_dir']
):        
    query_embed_model = load_embedder(query_embed_model_name, model_kwargs=query_embed_kwargs)
    chunk_embed_model = load_embedder(chunk_embed_model_name, model_kwargs=chunk_embed_kwargs)

    llm, qa_prompt_tmpl = load_llm_and_qa_tmpl(
        llm_model_name,
        max_new_tokens,
        cache_dir
    )

    retriever = load_retriever(
        vector_index_path,
        chunk_embed_model,
        k_vector_search,
        article_path,
        bm25_index_path,
        k_text_search
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        embed_model=query_embed_model,
        llm=llm,
        text_qa_template=qa_prompt_tmpl,
    )

    return query_engine


query_engine = creating_query_engine()

def query_rag_system(query):
    return str(query_engine.query(query))


if __name__ == "__main__":
    while True:
        try:
            query = input("\nYour question: ")
            print(query_rag_system(query))
        except Exception as e:
            print(f"An error occurred: {e}")
