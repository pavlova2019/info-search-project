from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel

from util.key_imports import *


class EmbeddingModel(Embeddings, BaseModel):
    tokenizer: Any
    device: str
    model: Any
    max_length: int
    model_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(kwargs["model_path"])
        self.device = kwargs["device"]
        self.model = AutoModel.from_pretrained(kwargs["model_path"]).to(self.device)
        self.max_length = kwargs["max_length"]

    @property
    def _llm_type(self) -> str:
        return self.version

    def _encode(self, text):
        pass

    def embed_documents(self, text: str) -> List[EmbeddingVector]:
        pass

    def embed_query(self, text: str) -> EmbeddingVector:
        pass
