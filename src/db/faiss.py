from faiss import Index, IndexFlatL2

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.indices import VectorStoreIndex

from util.key_imports import *

DEFAULT_INDEX = IndexFlatL2

class FaissVectorStoreIndex(VectorStoreIndex):
    def __init__(self, nodes, d: int, faiss_index: Index = DEFAULT_INDEX) -> None:
        super.__init__()
        self.faiss_index = faiss_index(d=d)
        pass

