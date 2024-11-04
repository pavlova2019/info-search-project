import torch
from tqdm import tqdm
from transformers import (
    PreTrainedModel, PreTrainedTokenizer
)


@torch.no_grad()
def get_embeddings(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        texts: list[str],
        batch_size: int = 32,
        truncation: bool = True,
        max_length: int = 512
) -> torch.Tensor:
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc='Processing batches'):
        batch_texts = texts[i:i + batch_size]

        encoded_input = tokenizer(batch_texts, return_tensors='pt', padding=True,
                                  truncation=truncation, max_length=max_length).to(model.device)

        batch_embeddings = model(**encoded_input)['pooler_output']
        all_embeddings.append(batch_embeddings)

    return torch.cat(all_embeddings, dim=0)
