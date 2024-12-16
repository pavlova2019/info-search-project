from typing import Optional
from src.util.article import Article
import src.config as cfg

import pandas as pd


def load_articles(filepath: str = cfg.TEST_DATASET, n: Optional[int] = None):
    dataset_small = pd.read_json(filepath, lines=True)
    n = len(dataset_small) if not n else n
    return [Article(
        text=dataset_small["article_text"][i],
        article_id=dataset_small["article_id"][i],
        abstract_text=dataset_small["abstract_text"][i]
    ) for i in range(n)]
