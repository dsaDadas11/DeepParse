import argparse
import subprocess
import sys
from pathlib import Path

from sqlalchemy import text

from runtime_config import ALLOWED_UPLOAD_EXTENSIONS
from service.core.file_parse import execute_insert_process
from service.core.rag.utils.es_conn import ESConnection
from utils.database import get_session_factory


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_SINGLE_FILE_TIMEOUT_SECONDS = 900


def gather_files(source_dir: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in source_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS
    )


def reset_user_state(user_id: str) -> None:
    es_connection = ESConnection()
    es_connection.es.indices.delete(index=user_id, ignore_unavailable=True)
    print(f"[reset] deleted index: {user_id}", flush=True)

    session_factory = get_session_factory()
    db = session_factory()
    try:
        db.execute(
            text(
                """
                DELETE FROM messages
                WHERE session_id IN (
                    SELECT session_id
                    FROM sessions
                    WHERE user_id = :user_id
                )
                """
            ),
            {"user_id": user_id},
        )
        db.execute(
            text("DELETE FROM sessions WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        db.execute(
            text("DELETE FROM knowledgebases WHERE user_id = :user_id"),
            {"user_id": user_id},
        )
        db.commit()
        print(f"[reset] cleared database rows for: {user_id}", flush=True)
    finally:
        db.close()


def reset_user_index(user_id: str) -> None:
    es_connection = ESConnection()
    es_connection.es.indices.delete(index=user_id, ignore_unavailable=True)
    print(f"[reset_index_only] deleted index: {user_id}", flush=True)


def get_completed_files(user_id: str) -> set[str]:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        rows = db.execute(
            text(
                """
                SELECT file_name
                FROM knowledgebases
                WHERE user_id = :user_id
                """
            ),
            {"user_id": user_id},
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        db.close()


def insert_knowledgebase_row(user_id: str, file_name: str) -> None:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        existing = db.execute(
            text(
                """
                SELECT 1
                FROM knowledgebases
                WHERE user_id = :user_id AND file_name = :file_name
                LIMIT 1
                """
            ),
            {
                "user_id": user_id,
                "file_name": file_name,
            },
        ).fetchone()
        if existing:
            return

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
        db.commit()
    finally:
        db.close()


def process_single_file(user_id: str, file_path: Path) -> int:
    chunk_count = execute_insert_process(str(file_path), file_path.name, user_id)
    insert_knowledgebase_row(user_id, file_path.name)
    print(f"[done] {file_path.name} chunks={chunk_count}", flush=True)
    return chunk_count


def run_single_file_subprocess(
    user_id: str,
    file_path: Path,
    timeout_seconds: int = DEFAULT_SINGLE_FILE_TIMEOUT_SECONDS,
) -> int:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--user-id",
        user_id,
        "--file-path",
        str(file_path),
        "--single-file",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", file=sys.stderr, flush=True)
        raise RuntimeError(
            f"single-file rebuild timed out after {timeout_seconds}s"
        ) from exc
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"single-file rebuild failed with exit code {result.returncode}")
    return 0


def rebuild_user_corpus(
    user_id: str,
    source_dir: Path,
    *,
    reset_first: bool,
    reset_index_only: bool,
    resume: bool,
    continue_on_error: bool,
    single_file_timeout_seconds: int,
) -> None:
    files = gather_files(source_dir)
    if not files:
        raise RuntimeError(f"No supported files found in {source_dir}")

    completed = get_completed_files(user_id)
    if reset_first:
        reset_user_state(user_id)
        completed = set()
    elif reset_index_only:
        reset_user_index(user_id)
        completed = set()
    elif completed and not resume:
        raise RuntimeError(
            "Existing knowledgebase rows found. Use --resume to continue or --reset-first to rebuild from scratch."
        )

    print(
        f"[start] user_id={user_id} files={len(files)} completed={len(completed)} source_dir={source_dir}",
        flush=True,
    )

    succeeded = 0
    failed: list[tuple[str, str]] = []

    for index, file_path in enumerate(files, start=1):
        if file_path.name in completed:
            print(f"[skip] {index}/{len(files)} {file_path.name}", flush=True)
            continue

        print(f"[index] {index}/{len(files)} {file_path.name}", flush=True)
        try:
            run_single_file_subprocess(
                user_id,
                file_path,
                timeout_seconds=single_file_timeout_seconds,
            )
            completed.add(file_path.name)
            succeeded += 1
        except Exception as exc:
            failed.append((file_path.name, str(exc)))
            print(f"[fail] {file_path.name} error={exc}", flush=True)
            if not continue_on_error:
                break

    print(
        f"[summary] completed={len(completed)} total={len(files)} newly_succeeded={succeeded} failed={len(failed)}",
        flush=True,
    )
    if failed:
        for file_name, error in failed:
            print(f"[failed_file] {file_name} -> {error}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--file-path")
    parser.add_argument("--reset-first", action="store_true")
    parser.add_argument("--reset-index-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--single-file", action="store_true")
    parser.add_argument(
        "--single-file-timeout-seconds",
        type=int,
        default=DEFAULT_SINGLE_FILE_TIMEOUT_SECONDS,
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.reset_first and args.reset_index_only:
        parser.error("--reset-first and --reset-index-only cannot be used together")

    if args.single_file:
        if not args.file_path:
            parser.error("--single-file requires --file-path")
        process_single_file(args.user_id, Path(args.file_path))
        return

    if not args.source_dir:
        parser.error("full rebuild mode requires --source-dir")

    rebuild_user_corpus(
        args.user_id,
        Path(args.source_dir),
        reset_first=args.reset_first,
        reset_index_only=args.reset_index_only,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
        single_file_timeout_seconds=args.single_file_timeout_seconds,
    )


if __name__ == "__main__":
    main()
