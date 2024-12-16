import faiss
from llama_index.core.node_parser import SentenceSplitter
from src.embedders.embedder_config import models_config
import src.config as cfg


chunk_size: int = 512
overlap: int = 50

def get_transformations():
    return [
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap),
    ]

def get_faiss_index(embedding_dim: int = models_config[cfg.embed_model_name]['embedding_dim']):
    return faiss.IndexFlat(embedding_dim)
