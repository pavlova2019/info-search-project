from util.article import Article

import pandas as pd


def load_articles(n: int = 1):
    dataset_small = pd.read_json("data/small_dataset.json", lines=True)
    return [Article(
        text=dataset_small["article_text"][i],
        article_id=dataset_small["article_id"][i],
        abstract_text=dataset_small["abstract_text"][i]
    ) for i in range(n)]
