import pandas as pd
from typing import Optional
from src.util.article import Article, Timestamp
import numpy as np


def load_articles(filepath: str, n: Optional[int] = None):
    data = pd.read_json(filepath, lines=True)
    data = data[:10]
    n = len(data) if not n else n
    return data.apply(create_article, axis=1, raw=True)


def create_article(row: np.ndarray):
    return Article(
        article_id=row[0],
        text="\n\n".join(key + "\n\n" + val for key,
                         val in row[18].items() if key != "title_text"),
        published=Timestamp(row[5]),
        title=row[7],
        authors=[dct["name"] for dct in row[11]],
        category=row[19],
        tags=[dct["term"] for dct in row[16]]
    )
