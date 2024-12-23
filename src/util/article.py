from src.util.arxiv_url_type import ArticleId
from llama_index.core import Document

from pandas import Timestamp

class Article(Document):

    def __init__(
            self,
            text: str,
            article_id: str,
            published: Timestamp,
            title: str,
            # authors: list[str],
            category: str,
            tags: list[str]
            ):
        super().__init__(
            text=f"{text}",
            metadata={
                # "id": article_id,
                "published": published,
                "title": title,
                # "authors": authors,
                "category": category,
                "tags": tags
            },
            excluded_llm_metadata_keys=["tags"],
            excluded_embed_metadata_keys=["published", "category", "tags"],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        self.id_ = ArticleId(article_id)

    
    @classmethod
    def class_name(cls) -> str:
        return "Article"
    
    @classmethod
    def example(cls, unique_num: int) -> Document:
        return Article(
            text="This article is the coolest thing in the world!" * 100,
            article_id="https://arxiv.org/abs/cool_" + str(unique_num),
            published=Timestamp(0),
            title="Cool Article #" + str(unique_num),
            # authors=["Cool Guy"],
            category="",
            tags=[]
        )
    