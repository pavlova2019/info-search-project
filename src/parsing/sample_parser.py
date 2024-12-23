import pandas as pd
from typing import Optional
from src.util.article import Article, Timestamp
import numpy as np


def load_articles(filepath: str, n: Optional[int] = None):
    data = pd.read_json(filepath)
    n = len(data) if not n else n
    data = data.iloc[:n]
    return data.apply(create_article, axis=1, raw=True)


def create_article(row: np.ndarray):
    return Article(
        article_id=row[0],
        text="\n\n".join(key + "\n\n" + val for key,
                         val in row[18].items() if key != "title_text"),
        published="-".join(str(i) for i in row[6][2::-1]),
        title=row[7],
        authors=[dct["name"] for dct in row[11][:5]],
        category=row[19],
        tags=[dct["term"] for dct in row[16]]
    )
    