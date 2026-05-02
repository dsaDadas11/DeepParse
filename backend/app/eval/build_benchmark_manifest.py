import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from eval.benchmark_versions import (
    ANSWER_RULE_VERSION,
    CHUNK_VERSION,
    CORPUS_VERSION,
    EVAL_SCRIPT_VERSION,
    RETRIEVER_VERSION,
)
from service.core.file_parse import build_parser_config
from service.core.rag.utils.es_conn import ESConnection
from service.core.rag_config import (
    DENSE_SIMILARITY_FALLBACK,
    DEFAULT_CHUNK_DELIMITER,
    DEFAULT_CHUNK_TOKEN_NUM,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LAYOUT_RECOGNIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_USE_INTENT_BONUS,
    DEFAULT_USE_MODEL_RERANK,
    DEFAULT_VECTOR_SIMILARITY_WEIGHT,
    FUSION_WEIGHT_TEXT,
    FUSION_WEIGHT_VECTOR,
    PDF_PLAIN_TEXT_SIZE_THRESHOLD_BYTES,
    RERANK_CANDIDATE_CAP,
    RERANK_PAGE_LIMIT,
    RERANK_TOKEN_CAP,
    RUNTIME_SCORE_LEGAL_METADATA_STRONG_BONUS,
    RUNTIME_SCORE_LEGAL_METADATA_UNIT,
)
from runtime_config import (
    ENABLE_LEGAL_METADATA_HARD_FILTER,
    ENABLE_LEGAL_METADATA_ROUTE,
    ENABLE_LEGAL_METADATA_SCORING,
    RETRIEVAL_ROUTE_MODE,
)
from utils.database import get_session_factory


DEFAULT_RETRIEVAL_CASES = Path(__file__).with_name("resume_retrieval_benchmark_manual_v2.json")
DEFAULT_GENERATION_CASES = Path(__file__).with_name("resume_generation_benchmark_manual_v2.json")
DEFAULT_OUTPUT = Path(__file__).with_name("benchmark_manifest.json")


def iso_utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(data) -> str:
    return sha256_text(json.dumps(data, ensure_ascii=False, sort_keys=True))


def load_json(path: Path):
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def case_type_counts(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        case_type = case.get("case_type", "unknown")
        counts[case_type] = counts.get(case_type, 0) + 1
    return dict(sorted(counts.items()))


def build_file_snapshot(path: Path, root: Path | None = None) -> dict:
    stats = path.stat()
    relative_path = path.relative_to(root).as_posix() if root else path.name
    return {
        "path": relative_path,
        "size_bytes": stats.st_size,
        "mtime_utc": dt.datetime.fromtimestamp(stats.st_mtime, tz=dt.timezone.utc).replace(microsecond=0).isoformat(),
        "sha256": sha256_file(path),
    }


def build_corpus_snapshot(source_dir: Path) -> dict:
    if not source_dir.exists():
        return {
            "source_dir": str(source_dir),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "aggregate_hash": stable_hash([]),
            "files": [],
        }

    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    snapshots = [build_file_snapshot(path, source_dir) for path in files]
    return {
        "source_dir": str(source_dir),
        "exists": True,
        "file_count": len(snapshots),
        "total_bytes": sum(item["size_bytes"] for item in snapshots),
        "aggregate_hash": stable_hash(
            [
                {
                    "path": item["path"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                }
                for item in snapshots
            ]
        ),
        "files": snapshots,
    }


def build_chunk_snapshot(source_dir: Path) -> dict:
    if not source_dir.exists():
        return {
            "defaults": {
                "chunk_token_num": DEFAULT_CHUNK_TOKEN_NUM,
                "delimiter": DEFAULT_CHUNK_DELIMITER,
                "layout_recognize": DEFAULT_LAYOUT_RECOGNIZE,
                "pdf_plain_text_threshold_bytes": PDF_PLAIN_TEXT_SIZE_THRESHOLD_BYTES,
            },
            "per_file_config_hash": stable_hash([]),
            "layout_summary": {},
            "files": [],
        }

    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    entries: list[dict] = []
    layout_summary: dict[str, int] = {}
    for path in files:
        parser_config = build_parser_config(str(path))
        layout = str(parser_config.get("layout_recognize", "unknown"))
        layout_summary[layout] = layout_summary.get(layout, 0) + 1
        entries.append(
            {
                "path": path.relative_to(source_dir).as_posix(),
                "parser_config": parser_config,
            }
        )

    return {
        "defaults": {
            "chunk_token_num": DEFAULT_CHUNK_TOKEN_NUM,
            "delimiter": DEFAULT_CHUNK_DELIMITER,
            "layout_recognize": DEFAULT_LAYOUT_RECOGNIZE,
            "pdf_plain_text_threshold_bytes": PDF_PLAIN_TEXT_SIZE_THRESHOLD_BYTES,
        },
        "per_file_config_hash": stable_hash(entries),
        "layout_summary": dict(sorted(layout_summary.items())),
        "files": entries,
    }


def build_evalset_snapshot(path: Path) -> dict:
    cases = load_json(path) or []
    return {
        "path": str(path),
        "cases": len(cases),
        "hash": stable_hash(cases),
        "case_type_counts": case_type_counts(cases),
    }


def get_knowledgebase_count(user_id: str) -> int | None:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        return db.execute(
            text("SELECT count(1) FROM knowledgebases WHERE user_id = :user_id"),
            {"user_id": user_id},
        ).scalar_one()
    except Exception:
        return None
    finally:
        db.close()


def get_chunk_count(user_id: str) -> int | None:
    try:
        es_connection = ESConnection()
        response = es_connection.es.count(index=user_id)
        return int(response.get("count", 0))
    except Exception:
        return None


def top_level_summary(report: dict | None, keys: tuple[str, ...]) -> dict | None:
    if not report:
        return None
    summary = report
    if isinstance(summary, dict) and "summary" in summary:
        summary = summary["summary"]
    if not isinstance(summary, dict):
        return None
    return {key: summary.get(key) for key in keys if key in summary}


def build_script_hashes(eval_dir: Path, app_dir: Path) -> dict[str, str]:
    script_paths = {
        "retrieval_py": app_dir / "service" / "core" / "retrieval.py",
        "search_v2_py": app_dir / "service" / "core" / "rag" / "nlp" / "search_v2.py",
        "file_parse_py": app_dir / "service" / "core" / "file_parse.py",
        "chat_py": app_dir / "service" / "core" / "chat.py",
        "answering_rules_py": app_dir / "service" / "core" / "answering_rules.py",
        "run_retrieval_compare_py": eval_dir / "run_retrieval_compare.py",
        "run_generation_eval_py": eval_dir / "run_generation_eval.py",
        "build_retrieval_eval_cases_py": eval_dir / "build_retrieval_eval_cases.py",
        "build_generation_eval_cases_py": eval_dir / "build_generation_eval_cases.py",
        "build_benchmark_manifest_py": eval_dir / "build_benchmark_manifest.py",
        "build_error_appendix_py": eval_dir / "build_error_appendix.py",
    }
    return {name: sha256_file(path) for name, path in script_paths.items() if path.exists()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--source-dir")
    parser.add_argument("--retrieval-cases", default=str(DEFAULT_RETRIEVAL_CASES))
    parser.add_argument("--generation-cases", default=str(DEFAULT_GENERATION_CASES))
    parser.add_argument("--retrieval-report")
    parser.add_argument("--generation-report")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source_dir = Path(args.source_dir) if args.source_dir else ROOT_DIR / "service" / "core" / "storage" / "file" / args.user_id
    retrieval_cases_path = Path(args.retrieval_cases)
    generation_cases_path = Path(args.generation_cases)
    retrieval_report_path = Path(args.retrieval_report) if args.retrieval_report else None
    generation_report_path = Path(args.generation_report) if args.generation_report else None
    output_path = Path(args.output)

    retrieval_evalset = build_evalset_snapshot(retrieval_cases_path)
    generation_evalset = build_evalset_snapshot(generation_cases_path)
    evalset_version = f"evalset_r{retrieval_evalset['cases']}_g{generation_evalset['cases']}"

    benchmark_tag = "+".join(
        [
            CORPUS_VERSION,
            CHUNK_VERSION,
            RETRIEVER_VERSION,
            ANSWER_RULE_VERSION,
            EVAL_SCRIPT_VERSION,
            evalset_version,
        ]
    )

    eval_dir = Path(__file__).resolve().parent
    app_dir = ROOT_DIR
    corpus_snapshot = build_corpus_snapshot(source_dir)
    chunk_snapshot = build_chunk_snapshot(source_dir)
    script_hashes = build_script_hashes(eval_dir, app_dir)

    retrieval_report = load_json(retrieval_report_path) if retrieval_report_path else None
    generation_report = load_json(generation_report_path) if generation_report_path else None

    runtime_summary = {
        "knowledgebase_files": get_knowledgebase_count(args.user_id),
        "chunk_count": get_chunk_count(args.user_id),
        "retrieval": {
            "current": top_level_summary(
                retrieval_report.get("current") if isinstance(retrieval_report, dict) else None,
                (
                    "cases",
                    "hit_at_1",
                    "hit_at_3",
                    "hit_at_k",
                    "mrr",
                    "evidence_cases",
                    "evidence_hit_at_1",
                    "evidence_hit_at_3",
                    "evidence_hit_at_k",
                    "avg_latency_ms",
                    "p95_latency_ms",
                ),
            ),
            "weak_baseline": top_level_summary(
                retrieval_report.get("weak_baseline") if isinstance(retrieval_report, dict) else None,
                (
                    "cases",
                    "hit_at_1",
                    "hit_at_3",
                    "hit_at_k",
                    "mrr",
                    "evidence_cases",
                    "evidence_hit_at_1",
                    "evidence_hit_at_3",
                    "evidence_hit_at_k",
                    "avg_latency_ms",
                    "p95_latency_ms",
                ),
            ),
            "delta": top_level_summary(
                retrieval_report.get("delta") if isinstance(retrieval_report, dict) else None,
                (
                    "hit_at_1",
                    "hit_at_3",
                    "hit_at_k",
                    "mrr",
                    "evidence_hit_at_1",
                    "evidence_hit_at_3",
                    "evidence_hit_at_k",
                ),
            ),
        },
        "generation": top_level_summary(
            generation_report,
            (
                "cases",
                "answerable_cases",
                "abstain_cases",
                "answer_accuracy",
                "citation_support_rate",
                "grounded_answer_rate",
                "abstain_success_rate",
                "hallucination_rate",
                "rule_answer_rate",
                "model_answer_rate",
                "avg_latency_ms",
                "p95_latency_ms",
            ),
        ),
    }

    retrieval_config = {
        "top_k": DEFAULT_TOP_K,
        "embed_model": DEFAULT_EMBED_MODEL,
        "vector_similarity_weight": DEFAULT_VECTOR_SIMILARITY_WEIGHT,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "rerank_page_limit": RERANK_PAGE_LIMIT,
        "rerank_candidate_cap": RERANK_CANDIDATE_CAP,
        "rerank_token_cap": RERANK_TOKEN_CAP,
        "fusion_weight_text": FUSION_WEIGHT_TEXT,
        "fusion_weight_vector": FUSION_WEIGHT_VECTOR,
        "dense_similarity_fallback": DENSE_SIMILARITY_FALLBACK,
        "use_model_rerank": DEFAULT_USE_MODEL_RERANK,
        "use_intent_bonus": DEFAULT_USE_INTENT_BONUS,
        "retrieval_route_mode": RETRIEVAL_ROUTE_MODE,
        "enable_legal_metadata_route": ENABLE_LEGAL_METADATA_ROUTE,
        "enable_legal_metadata_scoring": ENABLE_LEGAL_METADATA_SCORING,
        "enable_legal_metadata_hard_filter": ENABLE_LEGAL_METADATA_HARD_FILTER,
        "legal_metadata_score_unit": RUNTIME_SCORE_LEGAL_METADATA_UNIT,
        "legal_metadata_strong_bonus": RUNTIME_SCORE_LEGAL_METADATA_STRONG_BONUS,
    }

    fingerprint = stable_hash(
        {
            "benchmark_tag": benchmark_tag,
            "corpus_hash": corpus_snapshot["aggregate_hash"],
            "chunk_hash": chunk_snapshot["per_file_config_hash"],
            "retrieval_config": retrieval_config,
            "script_hashes": script_hashes,
            "retrieval_evalset_hash": retrieval_evalset["hash"],
            "generation_evalset_hash": generation_evalset["hash"],
        }
    )

    manifest = {
        "generated_at": iso_utc_now(),
        "user_id": args.user_id,
        "benchmark_tag": benchmark_tag,
        "benchmark_fingerprint": fingerprint,
        "components": {
            "corpus_version": CORPUS_VERSION,
            "chunk_version": CHUNK_VERSION,
            "retriever_version": RETRIEVER_VERSION,
            "answer_rule_version": ANSWER_RULE_VERSION,
            "eval_version": EVAL_SCRIPT_VERSION,
            "evalset_version": evalset_version,
        },
        "paths": {
            "source_dir": str(source_dir),
            "retrieval_cases": str(retrieval_cases_path),
            "generation_cases": str(generation_cases_path),
            "retrieval_report": str(retrieval_report_path) if retrieval_report_path else None,
            "generation_report": str(generation_report_path) if generation_report_path else None,
        },
        "corpus": corpus_snapshot,
        "chunk_config": chunk_snapshot,
        "retrieval_config": retrieval_config,
        "code_hashes": script_hashes,
        "eval": {
            "retrieval_cases": retrieval_evalset,
            "generation_cases": generation_evalset,
        },
        "runtime_summary": runtime_summary,
    }

    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
