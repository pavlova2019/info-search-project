from time import sleep
from typing import List

from embedders.sample_embedder import EmbeddingModel
from text_gen.sample_model import LanguageModel
from util.article import Article
from parsing.sample_parser import load_articles

from faiss import IndexHNSWFlat

from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.langchain import LangChainLLM
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore

# placeholders
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import MockLLM

articles: List[Article] = load_articles(10)

embedding_dim = 16
M = 4

faiss_index = IndexHNSWFlat(embedding_dim, M)
faiss_index.verbose = True
vector_store = FaissVectorStore(faiss_index=faiss_index)
vector_store.stores_text = True


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

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents=articles,
    transformations=[
        splitter,
        embedder_model
    ],
    embed_model=embedder_model,
    storage_context=storage_context,
    store_nodes_override=True
)
query_engine = index.as_query_engine(llm=llm_model)

response = query_engine.query("Which article is the coolest?")
print(response)


