"""Low-level retrieval helpers used by runtime fallback and chunk expansion."""

from service.core.rag.nlp.search_v2 import Dealer
from service.core.rag.utils.es_conn import ESConnection
from service.core.rag_config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_SIMILARITY_WEIGHT,
)

es_connection = ESConnection()
dealer = Dealer(dataStore=es_connection)


def _build_message(chunk: dict, index_name: str, position: int) -> dict:
    docnm = chunk.get("docnm_kwd", "N/A").split("/")[-1]
    return {
        "id": position,
        "index_name": index_name,
        "document_id": chunk.get("doc_id", "N/A"),
        "chunk_id": chunk.get("chunk_id"),
        "page_num": chunk.get("page_num_int"),
        "document_name": docnm,
        "content_with_weight": chunk.get("content_with_weight", "N/A"),
    }


def retrieve_content(indexNames: str, question: str, top_k: int = DEFAULT_TOP_K):
    results = dealer.retrieval(
        question=question,
        embd_mdl=DEFAULT_EMBED_MODEL,
        tenant_ids=indexNames,
        kb_ids=None,
        vector_similarity_weight=DEFAULT_VECTOR_SIMILARITY_WEIGHT,
        page=1,
        page_size=top_k,
    )

    extracted_data = []
    for i, chunk in enumerate(results["chunks"], start=1):
        extracted_data.append(_build_message(chunk, indexNames, i))

    return extracted_data


def retrieve_document_chunks(index_name: str, document_id: str, limit: int = 24) -> list[dict]:
    if not index_name or not document_id or limit <= 0:
        return []

    response = es_connection.es.search(
        index=index_name,
        body={
            "query": {"term": {"doc_id": document_id}},
            "sort": [
                {"page_num_int": {"order": "asc", "unmapped_type": "integer"}},
                {"chunk_id": {"order": "asc", "unmapped_type": "keyword"}},
            ],
            "_source": [
                "chunk_id",
                "content_with_weight",
                "doc_id",
                "docnm_kwd",
                "page_num_int",
            ],
            "size": limit,
        },
        timeout="120s",
        track_total_hits=False,
    )

    hits = response.get("hits", {}).get("hits", [])
    return [
        _build_message(hit.get("_source", {}), index_name, i)
        for i, hit in enumerate(hits, start=1)
        if hit.get("_source")
    ]
