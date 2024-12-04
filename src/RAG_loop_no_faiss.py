from typing import List

from embedders.sample_embedder import EmbeddingModel
from text_gen.sample_model import LanguageModel
from util.article import Article
from parsing.sample_parser import load_articles

from llama_index.llms.langchain import LangChainLLM
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore

# placeholders
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

articles: List[Article] = load_articles(10)

embedding_dim = 16

# params_embed = {   # выкинуть в конфиг
#     "model_name": "...",
#     "model_path": "...",
#     "device": "cuda:0",
#     "max_length": embedding_dim
# }
# embedder_model = EmbeddingModel(**params_embed)

embedder_model = MockEmbedding(embed_dim=embedding_dim)

splitter = TokenTextSplitter(
    chunk_size=1024,
    chunk_overlap=20,
    separator=" "
)

pipeline = IngestionPipeline(
    transformations=[
        splitter,
        embedder_model,
    ],
)

nodes = pipeline.run(show_progress=True, documents=articles, num_workers=1)

# GENERATE_CONFIG = {   # выкинуть в конфиг
#     "max_length": 4096,
#     "temperature": 1e-15,
#     "repetition_penalty": 1.07
# }
# llm = LanguageModel(
#     model_path="...",
#     device="cuda:0",
#     generate_config=GENERATE_CONFIG
# )
# llm_model = LangChainLLM(llm=llm)

llm_model = MockLLM(max_tokens=32)

index = VectorStoreIndex.from_documents(
    documents=articles,
    transformations=[
        splitter,
        embedder_model
    ],
    embed_model=embedder_model,
)
query_engine = index.as_query_engine(llm=llm_model)

response = query_engine.query("Which article is the coolest?")
print(response)


