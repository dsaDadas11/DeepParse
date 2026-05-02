from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from database.knowledgebase_access import insert_knowledgebase
from service.core.file_parse import execute_insert_process
from service.document_operations import delete_indexed_chunks
from utils import logger
from utils.database import get_session_factory


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _serialize_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


@dataclass
class UploadTask:
    task_id: str
    user_id: str
    file_name: str
    file_path: str
    status: str
    message: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    indexed_chunks: int | None = None
    error: str | None = None
    retry_count: int = 0

    def to_response(self) -> dict:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "file_name": self.file_name,
            "status": self.status,
            "message": self.message,
            "created_at": _serialize_timestamp(self.created_at),
            "started_at": _serialize_timestamp(self.started_at),
            "finished_at": _serialize_timestamp(self.finished_at),
            "indexed_chunks": self.indexed_chunks,
            "error": self.error,
            "retry_count": self.retry_count,
        }


_TABLE_READY = False
_TABLE_LOCK = threading.Lock()
_TABLE_SETUP_LOCK_KEY = 4040313


def _row_to_task(row) -> UploadTask:
    return UploadTask(
        task_id=row.task_id,
        user_id=row.user_id,
        file_name=row.file_name,
        file_path=row.file_path,
        status=row.status,
        message=row.message,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
        indexed_chunks=row.indexed_chunks,
        error=row.error,
        retry_count=row.retry_count or 0,
    )


def _ensure_task_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return

    with _TABLE_LOCK:
        if _TABLE_READY:
            return

        session_factory = get_session_factory()
        db = session_factory()
        try:
            if getattr(db.bind.dialect, "name", "") == "postgresql":
                db.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_key)"),
                    {"lock_key": _TABLE_SETUP_LOCK_KEY},
                )
            db.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS upload_tasks (
                        task_id VARCHAR(16) PRIMARY KEY,
                        user_id VARCHAR(255) NOT NULL,
                        file_name VARCHAR(255) NOT NULL,
                        file_path TEXT NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        message TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP NULL,
                        finished_at TIMESTAMP NULL,
                        indexed_chunks INTEGER NULL,
                        error TEXT NULL,
                        retry_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
            )
            db.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_upload_tasks_user_status
                    ON upload_tasks(user_id, status)
                    """
                )
            )
            db.execute(
                text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_upload_tasks_created_at
                    ON upload_tasks(created_at)
                    """
                )
            )
            db.execute(
                text(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS uq_upload_tasks_active_file
                    ON upload_tasks(user_id, file_name)
                    WHERE status IN ('pending', 'running')
                    """
                )
            )
            db.execute(
                text(
                    """
                    UPDATE upload_tasks
                    SET status = 'failed',
                        message = 'Task interrupted before completion.',
                        error = COALESCE(error, 'Task interrupted before completion.'),
                        finished_at = COALESCE(finished_at, CURRENT_TIMESTAMP)
                    WHERE status IN ('pending', 'running')
                    """
                )
            )
            db.commit()
            _TABLE_READY = True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


def create_upload_task(user_id: str, file_name: str, file_path: str) -> dict:
    _ensure_task_table()
    task = UploadTask(
        task_id=uuid.uuid4().hex[:16],
        user_id=user_id,
        file_name=file_name,
        file_path=file_path,
        status="pending",
        message="Queued for parsing and indexing.",
        created_at=_utc_now(),
    )

    session_factory = get_session_factory()
    db = session_factory()
    try:
        db.execute(
            text(
                """
                INSERT INTO upload_tasks (
                    task_id,
                    user_id,
                    file_name,
                    file_path,
                    status,
                    message,
                    created_at,
                    retry_count
                ) VALUES (
                    :task_id,
                    :user_id,
                    :file_name,
                    :file_path,
                    :status,
                    :message,
                    :created_at,
                    :retry_count
                )
                """
            ),
            {
                "task_id": task.task_id,
                "user_id": task.user_id,
                "file_name": task.file_name,
                "file_path": task.file_path,
                "status": task.status,
                "message": task.message,
                "created_at": task.created_at,
                "retry_count": task.retry_count,
            },
        )
        db.commit()
        return task.to_response()
    except IntegrityError as exc:
        db.rollback()
        raise ValueError("Another upload task for the same file is already queued.") from exc
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_upload_task(task_id: str) -> UploadTask | None:
    _ensure_task_table()
    session_factory = get_session_factory()
    db = session_factory()
    try:
        row = db.execute(
            text(
                """
                SELECT
                    task_id,
                    user_id,
                    file_name,
                    file_path,
                    status,
                    message,
                    created_at,
                    started_at,
                    finished_at,
                    indexed_chunks,
                    error,
                    retry_count
                FROM upload_tasks
                WHERE task_id = :task_id
                """
            ),
            {"task_id": task_id},
        ).fetchone()
        return _row_to_task(row) if row else None
    finally:
        db.close()


def list_active_upload_names(user_id: str) -> set[str]:
    _ensure_task_table()
    session_factory = get_session_factory()
    db = session_factory()
    try:
        rows = db.execute(
            text(
                """
                SELECT file_name
                FROM upload_tasks
                WHERE user_id = :user_id
                  AND status IN ('pending', 'running')
                """
            ),
            {"user_id": user_id},
        ).fetchall()
        return {row.file_name for row in rows}
    finally:
        db.close()


def has_other_active_upload(user_id: str, file_name: str, *, exclude_task_id: str | None = None) -> bool:
    _ensure_task_table()
    session_factory = get_session_factory()
    db = session_factory()
    try:
        if exclude_task_id:
            row = db.execute(
                text(
                    """
                    SELECT 1
                    FROM upload_tasks
                    WHERE user_id = :user_id
                      AND file_name = :file_name
                      AND status IN ('pending', 'running')
                      AND task_id <> :exclude_task_id
                    LIMIT 1
                    """
                ),
                {
                    "user_id": user_id,
                    "file_name": file_name,
                    "exclude_task_id": exclude_task_id,
                },
            ).fetchone()
            return bool(row)

        row = db.execute(
            text(
                """
                SELECT 1
                FROM upload_tasks
                WHERE user_id = :user_id
                  AND file_name = :file_name
                  AND status IN ('pending', 'running')
                LIMIT 1
                """
            ),
            {
                "user_id": user_id,
                "file_name": file_name,
            },
        ).fetchone()
        return bool(row)
    finally:
        db.close()


def _normalize_task_updates(changes: dict[str, object]) -> dict[str, object]:
    allowed_columns = {
        "status",
        "message",
        "started_at",
        "finished_at",
        "indexed_chunks",
        "error",
        "retry_count",
    }
    return {key: value for key, value in changes.items() if key in allowed_columns}


def _update_task_in_db(db, task_id: str, **changes) -> None:
    updates = _normalize_task_updates(changes)
    if not updates:
        return

    assignments = ", ".join(f"{column} = :{column}" for column in updates)
    params = {"task_id": task_id, **updates}
    db.execute(
        text(f"UPDATE upload_tasks SET {assignments} WHERE task_id = :task_id"),
        params,
    )


def _update_task(task_id: str, **changes) -> UploadTask | None:
    _ensure_task_table()
    updates = _normalize_task_updates(changes)
    if not updates:
        return get_upload_task(task_id)

    session_factory = get_session_factory()
    db = session_factory()
    try:
        _update_task_in_db(db, task_id, **updates)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return get_upload_task(task_id)


def _ensure_knowledgebase_row(db, user_id: str, file_name: str) -> None:
    exists = db.execute(
        text(
            """
            SELECT 1
            FROM knowledgebases
            WHERE user_id = :user_id
              AND file_name = :file_name
            LIMIT 1
            """
        ),
        {"user_id": user_id, "file_name": file_name},
    ).fetchone()
    if not exists:
        insert_knowledgebase(db, user_id, file_name)


def retry_upload_task(task_id: str) -> UploadTask:
    task = get_upload_task(task_id)
    if task is None:
        raise ValueError("Upload task not found.")
    if task.status != "failed":
        raise ValueError("Only failed upload tasks can be retried.")
    if not task.file_path or not os.path.exists(task.file_path):
        raise ValueError("Original uploaded file is missing. Please upload the file again.")
    if has_other_active_upload(task.user_id, task.file_name, exclude_task_id=task.task_id):
        raise ValueError("Another upload task for the same file is already running.")

    try:
        retried = _update_task(
            task_id,
            status="pending",
            message="Queued for retry.",
            started_at=None,
            finished_at=None,
            indexed_chunks=None,
            error=None,
            retry_count=(task.retry_count or 0) + 1,
        )
    except IntegrityError as exc:
        raise ValueError("Another upload task for the same file is already running.") from exc
    if retried is None:
        raise ValueError("Upload task not found.")
    return retried


def process_upload_task(task_id: str) -> None:
    task = get_upload_task(task_id)
    if task is None:
        return

    _update_task(
        task_id,
        status="running",
        message="Parsing and indexing document.",
        started_at=_utc_now(),
        error=None,
    )

    indexed = False
    try:
        chunk_count = execute_insert_process(task.file_path, task.file_name, task.user_id)
        indexed = True
        session_factory = get_session_factory()
        db = session_factory()
        try:
            _ensure_knowledgebase_row(db, task.user_id, task.file_name)
            _update_task_in_db(
                db,
                task_id,
                status="success",
                message=f"Indexed {chunk_count} chunks.",
                indexed_chunks=chunk_count,
                finished_at=_utc_now(),
            )
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
    except Exception as exc:
        logger.exception("process_upload_task failed for %s", task.file_name)
        if indexed:
            try:
                delete_indexed_chunks(task.user_id, task.file_name)
            except Exception:
                logger.exception("delete_indexed_chunks rollback failed for %s", task.file_name)
        _update_task(
            task_id,
            status="failed",
            message="Indexing failed. Uploaded file kept for retry.",
            error=str(exc),
            finished_at=_utc_now(),
        )
