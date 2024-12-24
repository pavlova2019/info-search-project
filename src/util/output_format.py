import html
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeRelationship

def get_str_metadata(response: Response) -> str:
    html_titles = []
    for source_node in response.source_nodes:
        node_info = source_node.node.relationships[NodeRelationship.SOURCE]
        url = node_info.node_id
        title = node_info.metadata['title'].replace('\n', '')
        html_title = f'<a href="{url}">{title}</a>'
        if html_title not in html_titles:
            html_titles.append(html_title)        
    return '\n'.join(title for title in html_titles)

def preprocess_html(message: str) -> str:
    return html.escape(message, quote=False).replace("&lt;", "<")\
                                            .replace("&gt;", ">")\
                                            .replace("&amp;", "&")

def get_output(response: Response) -> str:
    titles = get_str_metadata(response)
    return preprocess_html(f"<b>Found papers:</b>\n{titles}\n\n{str(response).strip()}")
