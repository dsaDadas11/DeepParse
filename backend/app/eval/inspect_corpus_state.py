from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)
sys.path.insert(0, "/app")

from rebuild_user_corpus import gather_files, get_completed_files
from service.core.rag.utils.es_conn import ESConnection


def inspect_corpus_state(user_id: str, source_dir: Path) -> dict:
    files = gather_files(source_dir)
    source_names = [path.name for path in files]
    source_set = set(source_names)
    completed = get_completed_files(user_id)
    missing = [name for name in source_names if name not in completed]
    extra = sorted(completed - source_set)

    es = ESConnection().es
    index_exists = bool(es.indices.exists(index=user_id))
    doc_count = int(es.count(index=user_id)["count"]) if index_exists else 0

    kb_complete = len(source_names) > 0 and not missing and not extra
    ready = kb_complete and index_exists and doc_count > 0
    resume_safe = len(completed) > 0 and len(missing) > 0 and not extra and index_exists and doc_count > 0

    if not source_names:
        action = "error"
        reason = f"No supported files found in {source_dir}"
    elif ready:
        action = "skip"
        reason = "Corpus is already prepared."
    elif resume_safe:
        action = "resume"
        reason = "Partial rebuild detected; resume is safe."
    elif len(completed) == 0 and doc_count == 0 and not extra:
        action = "rebuild"
        reason = "No existing rebuild detected."
    else:
        action = "reset"
        reason = "Inconsistent corpus state detected; full rebuild required."

    return {
        "user_id": user_id,
        "source_dir": str(source_dir),
        "source_file_count": len(source_names),
        "completed_file_count": len(completed),
        "missing_file_count": len(missing),
        "extra_db_file_count": len(extra),
        "index_exists": index_exists,
        "doc_count": doc_count,
        "action": action,
        "reason": reason,
        "missing_files": missing,
        "extra_db_files": extra,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--source-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    state = inspect_corpus_state(args.user_id, Path(args.source_dir))
    print(json.dumps(state, ensure_ascii=False))


if __name__ == "__main__":
    main()
