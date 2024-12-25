class ArticleId(str):
    def __new__(cls, *value):
        if value:
            v0 = value[0]
            if not type(v0) is str:
                raise TypeError('Unexpected type for URL: "%s"' % type(v0))
            if not (v0.startswith('http://arxiv.org/abs/') or v0.startswith('https://arxiv.org/abs/')):
                raise ValueError('Passed string value "%s" is not an'
                                 ' "http*://arxiv.org/abs/" URL' % (v0,))

        return str.__new__(cls, *value)
