from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


def insert_knowledgebase(db: Session, user_id: str, file_name: str) -> None:
    db.execute(
        text(
            """
            INSERT INTO knowledgebases (user_id, file_name)
            VALUES (:user_id, :file_name)
            """
        ),
        {
            "user_id": user_id,
            "file_name": file_name,
        },
    )


def verify_user_knowledgebase(db: Session, user_id: str) -> None:
    try:
        query_result = db.execute(
            text("SELECT id FROM knowledgebases WHERE user_id = :user_id LIMIT 1"),
            {"user_id": user_id},
        ).fetchone()
        if not query_result:
            raise HTTPException(
                status_code=461,
                detail="You do not have your own knowledge base yet.",
            )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Database operation failed: {exc}",
        ) from exc
