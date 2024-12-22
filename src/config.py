import warnings
warnings.filterwarnings('ignore')

import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# environment variables
TG_TOKEN = os.environ.get('TG_TOKEN')

TEST_DATASET = os.environ.get('TEST_DATASET', './data/small_dataset.json')
RATINGS_DB_PATH = os.environ.get('RATINGS_DB_PATH', './data/db/ratings.db')
VECTOR_INDEX = os.environ.get('VECTOR_INDEX', './data/db/vector_index')

# default cache folder
cache_vars = ['TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'HF_HOME', 'LLAMA_INDEX_CACHE_DIR']
default_cache = './hfcache'

for var in cache_vars:
    if not os.environ.get(var):
        os.environ[var] = default_cache

CACHE_DIR = os.environ.get('HF_HOME')


# hyperparameters
max_new_tokens = 512
similarity_top_k = 5


LLM_MODEL = Literal[
    "nvidia/Llama3-ChatQA-2-8B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "Qwen/Qwen2.5-7B-Instruct",
]

llm_model_name: LLM_MODEL = "nvidia/Llama3-ChatQA-2-8B"


EMBED_MODEL = Literal[
    "jinaai/jina-embeddings-v3",
]

embed_model_name: EMBED_MODEL = "jinaai/jina-embeddings-v3"
query_embed_kwargs = {'default_task': 'retrieval.query'}
chunk_embed_kwargs = {'default_task': 'retrieval.passage'}
