from sentence_transformers import SentenceTransformer

from app.core.pg_client import PgClient
from app.service.db_queries import SIMILAR_CHUNKS


search_rag_tool = {
    "type": "function",
    "function": {
        "name": "search_rag",
        "description": (
            "Search the user's uploaded documents for information relevant "
            "to the user's question. Use this when the answer may be found "
            "in the user's documents or when you are unsure about the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A concise semantic search query containing only the "
                        "information you want to retrieve from the documents."
                    )
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

async def search_rag(query: str, 
               chat_id: int, 
               pg: PgClient, 
               embed_model: SentenceTransformer,
               top_k: int=3):
    """
    1. embed the query
    """
    embeds = embed_model.encode(query)
    similar = await pg.fetch(SIMILAR_CHUNKS, embeds, chat_id, top_k)
    return [
        {
            "text": c["chunk_text"],
            "filename": c["filename_original"],
            "chunk_index": c["chunk_index"],
            "score": c["similarity"],
        }
        for c in similar
    ]