from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional
import src.config as cfg

def load_embedder(model_name: cfg.EMBED_MODEL = cfg.embed_model_name,
                  model_kwargs: Optional[dict] = None):
    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
        device="cuda:0",
        model_kwargs=model_kwargs
    )
