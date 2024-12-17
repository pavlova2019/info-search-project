import faiss
from llama_index.core.node_parser import SentenceSplitter

chunk_size: int = 512
overlap: int = 50

def get_transformations():
    return [
        SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap),
    ]

def get_faiss_index(embedding_dim: int):
    return faiss.IndexFlat(embedding_dim)
