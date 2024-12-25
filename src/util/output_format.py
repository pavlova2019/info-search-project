import re
from typing import Literal
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeRelationship
import telegramify_markdown
from telegramify_markdown import customize

customize.markdown_symbol.head_level_1 = "📌"
customize.markdown_symbol.link = "🔗"
customize.strict_markdown = False
customize.cite_expandable = True


def get_str_metadata(response: Response,
                     mode: Literal['MARKDOWN_V2', 'HTML'] = 'MARKDOWN_V2') -> str:
    titles = []
    for source_node in response.source_nodes:
        node_info = source_node.node.relationships[NodeRelationship.SOURCE]
        url = node_info.node_id
        title = node_info.metadata['title'].replace('\n', '')
        if mode == 'HTML':
            title = f'<a href="{url}">{title}</a>'
        elif mode == 'MARKDOWN_V2':
            title = f'[{title}]({url})'
        else:
            raise ValueError(f'Not supported mode: {mode}')
        
        if title not in titles:
            titles.append(title)
            
    return '\n'.join(title for title in titles)
    

def get_output(response: Response) -> str:
    titles = get_str_metadata(response)
    response = str(response).strip()
    output = telegramify_markdown.markdownify(
        f"*Found papers*:\n{titles}\n\n{response}",
        max_line_length=None,
        normalize_whitespace=False
    )        
    return output
