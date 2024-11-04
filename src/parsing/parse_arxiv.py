import feedparser
import csv


def parse_arxiv(category: str,
                fields: list[str],
                subcategory: str = '',
                number: int = 1000) -> list[list]:

    avail_fields = ['publish_date', 'title', 'summary', 'authors',
                    'links', 'arxiv_primary_category', 'tags']

    for field in fields:
        if field not in avail_fields:
            raise ValueError(f'{field} is not in {avail_fields}')

    arxiv_url = 'http://export.arxiv.org/api/query?'
    subcategory = '*' if not subcategory else subcategory
    search_query = f'cat:{category}.{subcategory}'

    start = 0
    max_results = number if number <= 1000 else 1000
    delay = 5

    data = []
    while len(data) < number:
        query = f'{arxiv_url}search_query={search_query}&start={start}&max_results={max_results}'
        feed = feedparser.parse(query)

        if len(feed.entries) == 0:
            print(f'Only {len(data)} papers were found for query "{search_query}"!')
            break

        for field in feed.entries:
            data.append([field[f] for f in fields])

        start += max_results
        max_results = min(1000, abs(number - len(data)))

    return data


def parse_summary(categories: list[str],
                  number: int = 1000) -> list[tuple]:
    data = []
    for cat in categories:
        out = parse_arxiv(category=cat, fields=['summary'], number=number)
        out = [(cat, paper[0]) for paper in out]
        data.extend(out)
    return data


def save_parse_to_csv(data: list[tuple],
                      header: list = ['tag', 'summary'],
                      filename: str = 'output.csv') -> None:
    try:
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)

    except Exception as e:
        return f"An error occurred: {e}"


def main():
    parsing_categories = ['cs', 'econ', 'math', 'astro-ph', 'cond-mat',
                          'nlin', 'nucl-ex', 'physics', 'q-bio', 'q-fin']

    tag_summaries = parse_summary(parsing_categories)
    save_parse_to_csv(data=tag_summaries, filename='../../data/sample_summaries.csv')


if __name__ == "__main__":
    main()
