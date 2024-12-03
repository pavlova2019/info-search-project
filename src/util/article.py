Url = str # replace with url regular expression

from llama_index.core import Document

class Article(Document):

    def __init__(self, text: str, url: Url, name: str, tag: str):
        super().__init__(
            text=text,
            metadata={
                "name": name,
                "tag": tag,
            },
            excluded_llm_metadata_keys=["url"],
            metadata_seperator="::",
            metadata_template="{key}=>{value}",
            text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
        )
        self.id_ = url

    
    @classmethod
    def class_name(cls) -> str:
        return "Article"
    
    @classmethod
    def example(cls) -> Document:
        return Article(
            text="The coolest thing in the world is this article!",
            url="https://article/cool",
            name="Cool Name",
            tag="cool"
        )
