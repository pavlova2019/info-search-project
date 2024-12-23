from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, Dict, List

class CustomHuggingFaceEmbedding(HuggingFaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_metrics(self, execution_time: float) -> None:
        pass
        # print(execution_time)

    def _embed_with_retry(
        self,
        sentences: List[str],
        prompt_name: Optional[str] = None,
    ) -> List[List[float]]:
        start_time = time.time()
        try:
            return super()._embed_with_retry(sentences, prompt_name)
        finally:
            self._save_metrics(time.time() - start_time)
            

def load_embedder(model_name: str,
                  model_kwargs: Optional[Dict] = None):
    return CustomHuggingFaceEmbedding(
        model_name=model_name,
        trust_remote_code=True,
        device='cuda:0',
        model_kwargs=model_kwargs
    ) 
