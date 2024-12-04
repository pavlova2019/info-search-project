from util.article import Article


def load_articles(n: int = 1):
    return [Article.example(i) for i in range(n)]
