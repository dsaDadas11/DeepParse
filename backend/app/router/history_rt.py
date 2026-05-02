from typing import List
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from schemas.message import FilestResponse, SessionListResponse, SessionResponse
from service.document_operations import delete_document
from utils.database import get_db
from utils.request_context import get_current_user_id

router = APIRouter()


@router.get("/get_files", response_model=List[FilestResponse])
async def get_documents_by_user_id(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    rows = db.execute(
        text(
            """
            SELECT user_id, file_name, created_at, updated_at
            FROM knowledgebases
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            """
        ),
        {"user_id": user_id},
    ).fetchall()

    return [
        FilestResponse(
            user_id=row.user_id,
            file_name=row.file_name,
            created_at=row.created_at.isoformat(),
            updated_at=row.updated_at.isoformat(),
        )
        for row in rows
    ]


@router.delete("/delete_file/{file_name}")
async def delete_document_endpoint(
    file_name: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    result = delete_document(user_id, unquote(file_name), db)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["message"])
    return {"message": result["message"]}


@router.get("/get_messages")
async def get_messages_by_session_id(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    rows = db.execute(
        text(
            """
            SELECT m.message_id, m.session_id, m.user_question, m.model_answer,
                   m.documents, m.recommended_questions, m.think, m.created_at
            FROM messages AS m
            INNER JOIN sessions AS s
                ON s.session_id = m.session_id
            WHERE m.session_id = :session_id
              AND s.user_id = :user_id
            ORDER BY m.created_at ASC
            """
        ),
        {"session_id": session_id, "user_id": user_id},
    ).fetchall()

    return [
        {
            "message_id": row.message_id,
            "session_id": row.session_id,
            "user_question": row.user_question,
            "model_answer": row.model_answer,
            "documents": row.documents,
            "recommended_questions": row.recommended_questions,
            "think": row.think,
            "created_at": row.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for row in rows
    ]


@router.get("/get_sessions", response_model=SessionListResponse)
async def get_sessions_by_user_id(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    rows = db.execute(
        text(
            """
            SELECT session_id, session_name, user_id, created_at, updated_at
            FROM sessions
            WHERE user_id = :user_id
            ORDER BY updated_at DESC
            """
        ),
        {"user_id": user_id},
    ).fetchall()

    sessions = [
        SessionResponse(
            session_id=row.session_id,
            session_name=row.session_name,
            user_id=row.user_id,
            created_at=row.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            updated_at=row.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
        )
        for row in rows
    ]
    return {"user_id": user_id, "sessions": sessions}
