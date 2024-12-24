from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeRelationship

def get_str_metadata(response: Response) -> str:
    titles = []
    for source_node in response.source_nodes:
        node_info = source_node.node.relationships[NodeRelationship.SOURCE]
        url = node_info.node_id
        title = node_info.metadata['title']
        titles.append(f"[{title}]({url})")
        
    return '\n'.join(title for title in titles)
