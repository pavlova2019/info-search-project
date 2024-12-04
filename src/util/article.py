ArticleId = str # replace with article_id regular expression

from llama_index.core import Document

class Article(Document):

    def __init__(self, text: str, article_id: ArticleId, abstract_text: str):
        super().__init__(
            text=text,
            metadata={
                "abstract_text": abstract_text
            },
            # excluded_llm_metadata_keys=["article_id"],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        self.id_ = article_id

    
    @classmethod
    def class_name(cls) -> str:
        return "Article"
    
    @classmethod
    def example(cls, unique_num: int) -> Document:
        return Article(
            text="This article is the coolest thing in the world!" * 100,
            article_id="https://article/cool/" + str(unique_num),
            abstract_text="Cool abstract" * 10
        )
