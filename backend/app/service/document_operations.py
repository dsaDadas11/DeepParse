import os

from sqlalchemy import text
from sqlalchemy.orm import Session

from service.core.api.utils.file_utils import get_project_base_directory
from service.core.rag.utils.es_conn import ESConnection
from utils import logger


def delete_indexed_chunks(user_id: str, file_name: str) -> int:
    deleted_count = 0
    es_connection = ESConnection()
    try:
        if es_connection.es.indices.exists(index=user_id):
            response = es_connection.es.delete_by_query(
                index=user_id,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"docnm": file_name}},
                                {"term": {"docnm_kwd": file_name}},
                                {"wildcard": {"docnm_kwd": f"*{file_name}"}},
                            ],
                            "minimum_should_match": 1,
                        }
                    }
                },
                refresh=True,
                conflicts="proceed",
            )
            deleted_count = response.get("deleted", 0)
    except Exception:
        logger.exception("Failed to delete ES chunks for %s", file_name)
    return deleted_count


def delete_document(user_id: str, file_name: str, db: Session) -> dict:
    try:
        row = db.execute(
            text(
                """
                SELECT id
                FROM knowledgebases
                WHERE user_id = :user_id AND file_name = :file_name
                """
            ),
            {"user_id": user_id, "file_name": file_name},
        ).fetchone()
        if not row:
            return {"status": "error", "message": "Document not found"}

        deleted_count = delete_indexed_chunks(user_id, file_name)

        file_path = get_project_base_directory("storage", "file", user_id, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)

        db.execute(
            text(
                """
                DELETE FROM knowledgebases
                WHERE user_id = :user_id AND file_name = :file_name
                """
            ),
            {"user_id": user_id, "file_name": file_name},
        )
        db.commit()
        return {
            "status": "success",
            "message": f"Deleted document and {deleted_count} indexed chunk(s).",
        }
    except Exception as exc:
        db.rollback()
        logger.exception("delete_document failed")
        return {"status": "error", "message": f"Failed to delete document: {exc}"}
