import warnings
warnings.filterwarnings('ignore')

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# default cache folder
cache_vars = ['TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'HF_HOME', 'LLAMA_INDEX_CACHE_DIR']
default_cache = './hfcache'

for var in cache_vars:
    if not os.environ.get(var):
        os.environ[var] = default_cache
        
# available models
LLM_MODEL = Literal[
    "nvidia/Llama3-ChatQA-2-8B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-2-9b-it",
]

EMBED_MODEL = Literal[
    "jinaai/jina-embeddings-v3",
]
