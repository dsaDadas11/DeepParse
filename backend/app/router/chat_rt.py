import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from database.knowledgebase_access import verify_user_knowledgebase
from runtime_config import ALLOWED_UPLOAD_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES
from schemas.chat import ChatRequest, SessionResponse
from service.core.api.utils.file_utils import get_project_base_directory
from service.core.chat import get_chat_completion
from service.core.conversation import load_session_history, rewrite_question_with_history
from service.core.retrieval_runtime import build_retrieval_context
from service.upload_tasks import (
    create_upload_task,
    get_upload_task,
    has_other_active_upload,
    list_active_upload_names,
    process_upload_task,
    retry_upload_task,
)
from utils import logger
from utils.database import get_db
from utils.request_context import get_current_user_id

router = APIRouter()

TEXT_LIKE_UPLOAD_EXTENSIONS = {".txt", ".md", ".html", ".json"}
ZIP_BASED_UPLOAD_EXTENSIONS = {".docx", ".xlsx", ".pptx"}
OLE_BASED_UPLOAD_EXTENSIONS = {".xls", ".ppt"}
ZIP_SIGNATURES = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
OLE_SIGNATURE = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"


def _looks_like_text_payload(sample: bytes) -> bool:
    if not sample:
        return True
    if b"\x00" in sample:
        return False

    text_like_count = sum(
        1
        for byte in sample
        if byte in (9, 10, 13) or 32 <= byte <= 126 or byte >= 0x80
    )
    return text_like_count / len(sample) >= 0.9


def validate_upload_signature(file_name: str, sample: bytes) -> None:
    suffix = Path(file_name).suffix.lower()
    if not sample:
        return

    if suffix == ".pdf" and not sample.startswith(b"%PDF-"):
        raise ValueError("File content does not match the .pdf extension.")

    if suffix in ZIP_BASED_UPLOAD_EXTENSIONS and not any(sample.startswith(sig) for sig in ZIP_SIGNATURES):
        raise ValueError(f"File content does not match the {suffix} extension.")

    if suffix in OLE_BASED_UPLOAD_EXTENSIONS and not sample.startswith(OLE_SIGNATURE):
        raise ValueError(f"File content does not match the {suffix} extension.")

    if suffix in TEXT_LIKE_UPLOAD_EXTENSIONS and not _looks_like_text_payload(sample):
        raise ValueError(f"File content does not look like a valid {suffix} file.")


def build_upload_path(storage_dir: str, file_name: str) -> str:
    storage_root = Path(storage_dir).resolve()
    destination = (storage_root / file_name).resolve()
    if destination.parent != storage_root:
        raise HTTPException(status_code=400, detail="Invalid file name.")
    return str(destination)


def user_facing_upload_error(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        return detail if isinstance(detail, str) else "Upload validation failed."
    if isinstance(exc, ValueError):
        return str(exc)
    return "Upload failed while queuing the file."


async def save_upload_file(upload_file: UploadFile, destination: str, file_name: str) -> int:
    total_size = 0
    signature_checked = False
    try:
        with open(destination, "wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break

                if not signature_checked:
                    validate_upload_signature(file_name, chunk[:8192])
                    signature_checked = True

                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise ValueError(
                        f"File size cannot exceed {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
                    )

                buffer.write(chunk)

        if total_size == 0:
            raise ValueError("File content is empty.")

        return total_size
    finally:
        await upload_file.close()


def normalize_upload_name(upload_file: UploadFile) -> str:
    raw_name = (upload_file.filename or "").strip().replace("\\", "/")
    file_name = os.path.basename(raw_name)
    if not file_name:
        raise HTTPException(status_code=400, detail="File name cannot be empty.")
    if file_name in {".", ".."} or any(sep in file_name for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid file name.")
    if "\x00" in file_name:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    suffix = Path(file_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(ALLOWED_UPLOAD_EXTENSIONS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix or 'unknown'}. Allowed types: {allowed}.",
        )

    return file_name


@router.post("/create_session", response_model=SessionResponse)
async def create_session(user_id: str = Depends(get_current_user_id)):
    _ = user_id
    session_id = uuid.uuid4().hex[:16]
    return {
        "session_id": session_id,
        "status": "success",
        "message": "Session created successfully",
    }


@router.post("/chat_on_docs")
async def chat_on_docs(
    session_id: str = Query(...),
    request: ChatRequest = Body(..., description="User message"),
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    try:
        verify_user_knowledgebase(db, user_id)

        question = request.message.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        history_turns = load_session_history(db, session_id, user_id)
        standalone_query, rewrite_latency_ms = rewrite_question_with_history(question, history_turns)
        retrieval_context = build_retrieval_context(
            user_id,
            question,
            history_turns=history_turns,
            standalone_query=standalone_query,
        )
        references = retrieval_context["references"]
        retrieval_trace = dict(retrieval_context.get("trace") or {})
        retrieval_trace["history_turn_count"] = len(history_turns)
        retrieval_trace["rewrite_latency_ms"] = round(rewrite_latency_ms, 2)
        logger.info(
            "chat_on_docs retrieval question=%r standalone=%r planned_queries=%s",
            question,
            retrieval_context["standalone_query"],
            retrieval_context["planned_queries"],
        )
        return StreamingResponse(
            get_chat_completion(
                session_id,
                user_id,
                question,
                references,
                history_turns=history_turns,
                standalone_query=retrieval_context["standalone_query"],
                retrieval_trace=retrieval_trace,
            ),
            media_type="text/event-stream",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat_on_docs failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat request failed. Please retry later.",
        ) from exc


@router.post("/upload_files")
async def upload_files(
    background_tasks: BackgroundTasks,
    session_id: Optional[str] = Query(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    _ = session_id

    storage_dir = get_project_base_directory("storage", "file", user_id)
    os.makedirs(storage_dir, exist_ok=True)

    existing_rows = db.execute(
        text(
            """
            SELECT file_name
            FROM knowledgebases
            WHERE user_id = :user_id
            """
        ),
        {"user_id": user_id},
    ).fetchall()
    existing_files = {row.file_name for row in existing_rows}

    requested_names: list[str] = []
    duplicate_batch_files: set[str] = set()
    for file in files:
        file_name = normalize_upload_name(file)
        if file_name in requested_names:
            duplicate_batch_files.add(file_name)
        requested_names.append(file_name)

    if duplicate_batch_files:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate files in the same request: {', '.join(sorted(duplicate_batch_files))}",
        )

    duplicate_files = sorted({name for name in requested_names if name in existing_files})
    if duplicate_files:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate files already exist: {', '.join(duplicate_files)}",
        )

    active_uploads = list_active_upload_names(user_id)
    duplicate_active_files = sorted({name for name in requested_names if name in active_uploads})
    if duplicate_active_files:
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate files already queued: {', '.join(duplicate_active_files)}",
        )

    successful_files: list[str] = []
    failed_files: list[str] = []
    tasks: list[dict] = []

    for file in files:
        file_name = normalize_upload_name(file)
        file_path = build_upload_path(storage_dir, file_name)

        try:
            save_result = await save_upload_file(file, file_path, file_name)
            if save_result <= 0:
                raise ValueError("File content is empty.")
            task_info = create_upload_task(user_id, file_name, file_path)
            background_tasks.add_task(process_upload_task, task_info["task_id"])
            tasks.append(task_info)
            successful_files.append(file_name)
        except Exception as exc:
            if os.path.exists(file_path):
                os.remove(file_path)
            failed_files.append(f"{file_name}: {user_facing_upload_error(exc)}")
            if isinstance(exc, (HTTPException, ValueError)):
                logger.warning("upload_files rejected %s: %s", file_name, user_facing_upload_error(exc))
            else:
                logger.exception("upload_files failed for %s", file_name)

    if successful_files and not failed_files:
        return {
            "status": "success",
            "message": "All files uploaded and queued successfully.",
            "queued_files": successful_files,
            "tasks": tasks,
            "total_files": len(files),
        }

    if successful_files:
        return {
            "status": "partial_success",
            "message": f"Queued {len(successful_files)} file(s) and failed {len(failed_files)} file(s).",
            "queued_files": successful_files,
            "tasks": tasks,
            "failed_files": failed_files,
            "total_files": len(files),
        }

    raise HTTPException(
        status_code=400,
        detail={
            "status": "failed",
            "message": "All files failed to upload.",
            "failed_files": failed_files,
            "total_files": len(files),
        },
    )


@router.get("/upload_tasks/{task_id}")
async def get_upload_task_status(
    task_id: str,
    user_id: str = Depends(get_current_user_id),
):
    task = get_upload_task(task_id)
    if task is None or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Upload task not found.")
    return {
        "status": "success",
        "message": "Upload task retrieved successfully.",
        "task": task.to_response(),
    }


@router.post("/upload_tasks/{task_id}/retry")
async def retry_failed_upload_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    task = get_upload_task(task_id)
    if task is None or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Upload task not found.")

    if task.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed upload tasks can be retried.")

    if has_other_active_upload(user_id, task.file_name, exclude_task_id=task.task_id):
        raise HTTPException(
            status_code=409,
            detail="Another upload task for the same file is already running.",
        )

    try:
        retried_task = retry_upload_task(task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    background_tasks.add_task(process_upload_task, task_id)
    return {
        "status": "success",
        "message": "Upload task queued for retry.",
        "task": retried_task.to_response(),
    }
