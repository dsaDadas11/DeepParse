import argparse
import json
import os
import sys
import time
from pathlib import Path

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from service.core.rag.utils.es_conn import ESConnection
from service.core.retrieval_runtime import retrieve_content
from utils.database import get_session_factory


DEFAULT_CASES_PATH = Path(__file__).with_name("resume_retrieval_benchmark_manual_v2.json")

DOCUMENT_NAME_ALIASES: dict[str, str] = {}


def normalize_text(value: str) -> str:
    return "".join(str(value).lower().split())


def canonical_document_name(value: str) -> str:
    return DOCUMENT_NAME_ALIASES.get(str(value or ""), str(value or ""))


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def dedupe_documents(results: list[dict]) -> list[dict]:
    unique_results: list[dict] = []
    seen: set[str] = set()
    for item in results:
        key = normalize_text(canonical_document_name(item.get("document_name", "")))
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(item)
    return unique_results


def get_knowledgebase_count(user_id: str) -> int:
    session_factory = get_session_factory()
    db = session_factory()
    try:
        return db.execute(
            text("SELECT count(1) FROM knowledgebases WHERE user_id = :user_id"),
            {"user_id": user_id},
        ).scalar_one()
    finally:
        db.close()


def get_chunk_count(user_id: str) -> int:
    es_connection = ESConnection()
    response = es_connection.es.count(index=user_id)
    return int(response.get("count", 0))


def find_document_rank(results: list[dict], expected_documents: list[str]) -> int | None:
    expected = {normalize_text(canonical_document_name(name)) for name in expected_documents}
    for index, item in enumerate(results, start=1):
        if normalize_text(canonical_document_name(item.get("document_name", ""))) in expected:
            return index
    return None


def find_evidence_rank(
    results: list[dict],
    expected_any_phrases: list[str] | None,
    expected_all_phrases: list[str] | None,
) -> int | None:
    any_phrases = [normalize_text(item) for item in (expected_any_phrases or []) if item]
    all_phrases = [normalize_text(item) for item in (expected_all_phrases or []) if item]

    if not any_phrases and not all_phrases:
        return None

    for index, item in enumerate(results, start=1):
        content = normalize_text(item.get("content_with_weight", ""))
        has_any = True if not any_phrases else any(phrase in content for phrase in any_phrases)
        has_all = True if not all_phrases else all(phrase in content for phrase in all_phrases)
        if has_any and has_all:
            return index
    return None


def compute_metrics(case_results: list[dict], top_k: int) -> dict:
    total = len(case_results)
    latencies = [item["latency_ms"] for item in case_results]

    hit_at_1 = sum(1 for item in case_results if item["document_rank"] == 1)
    hit_at_3 = sum(1 for item in case_results if item["document_rank"] and item["document_rank"] <= 3)
    hit_at_k = sum(1 for item in case_results if item["document_rank"] and item["document_rank"] <= top_k)
    mrr = sum(1.0 / item["document_rank"] for item in case_results if item["document_rank"]) / total if total else 0.0

    evidence_cases = [item for item in case_results if item["evidence_expected"]]
    evidence_total = len(evidence_cases)
    evidence_hit_at_1 = sum(1 for item in evidence_cases if item["evidence_rank"] == 1)
    evidence_hit_at_3 = sum(1 for item in evidence_cases if item["evidence_rank"] and item["evidence_rank"] <= 3)
    evidence_hit_at_k = sum(1 for item in evidence_cases if item["evidence_rank"] and item["evidence_rank"] <= top_k)

    return {
        "cases": total,
        "hit_at_1": hit_at_1 / total if total else 0.0,
        "hit_at_3": hit_at_3 / total if total else 0.0,
        "hit_at_k": hit_at_k / total if total else 0.0,
        "mrr": mrr,
        "avg_latency_ms": sum(latencies) / total if total else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "max_latency_ms": max(latencies) if latencies else 0.0,
        "evidence_cases": evidence_total,
        "evidence_hit_at_1": evidence_hit_at_1 / evidence_total if evidence_total else 0.0,
        "evidence_hit_at_3": evidence_hit_at_3 / evidence_total if evidence_total else 0.0,
        "evidence_hit_at_k": evidence_hit_at_k / evidence_total if evidence_total else 0.0,
    }


def summarize_metrics(case_results: list[dict], top_k: int) -> dict:
    summary = compute_metrics(case_results, top_k)
    case_types = sorted({item.get("case_type", "unknown") for item in case_results})
    summary["by_case_type"] = {
        case_type: compute_metrics(
            [item for item in case_results if item.get("case_type", "unknown") == case_type],
            top_k,
        )
        for case_type in case_types
    }
    return summary


def runtime_config_snapshot() -> dict:
    keys = (
        "LEGAL_TERM_NORMALIZATION_ENABLED",
        "RETRIEVAL_ROUTE_MODE",
        "ENABLE_RETRIEVAL_TRACE",
        "ENABLE_FALLBACK_ROUTE",
        "ENABLE_LEGAL_METADATA_ROUTE",
        "ENABLE_LEGAL_METADATA_SCORING",
        "ENABLE_LEGAL_METADATA_HARD_FILTER",
        "STRICT_CITATION_BINDING",
        "CONFLICT_ABSTAIN_ENABLED",
    )
    return {key: os.getenv(key) for key in keys}


def evaluate(user_id: str, cases_path: Path, top_k: int) -> dict:
    cases = load_cases(cases_path)
    case_results: list[dict] = []

    for case in cases:
        question = case["question"]
        start = time.perf_counter()
        results = retrieve_content(user_id, question, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        results = results[:top_k]
        unique_results = dedupe_documents(results)

        document_rank = find_document_rank(unique_results, case.get("expected_documents", []))
        evidence_rank = find_evidence_rank(
            results,
            case.get("expected_any_phrases"),
            case.get("expected_all_phrases"),
        )

        case_results.append(
            {
                "id": case["id"],
                "case_type": case.get("case_type", "unknown"),
                "question": question,
                "document_rank": document_rank,
                "evidence_rank": evidence_rank,
                "evidence_expected": bool(case.get("expected_any_phrases") or case.get("expected_all_phrases")),
                "latency_ms": round(latency_ms, 2),
                "top_documents": [item.get("document_name", "") for item in results],
                "unique_top_documents": [item.get("document_name", "") for item in unique_results],
            }
        )

    summary = summarize_metrics(case_results, top_k)
    summary["knowledgebase_files"] = get_knowledgebase_count(user_id)
    summary["chunk_count"] = get_chunk_count(user_id)
    summary["top_k"] = top_k

    return {
        "summary": summary,
        "cases": case_results,
        "config_snapshot": runtime_config_snapshot(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = evaluate(args.user_id, Path(args.cases), args.top_k)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
