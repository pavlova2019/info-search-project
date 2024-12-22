from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, Dict

def load_embedder(model_name: str,
                  model_kwargs: Optional[Dict] = None):
    return HuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
        device='cuda:0',
        model_kwargs=model_kwargs
    ) 
