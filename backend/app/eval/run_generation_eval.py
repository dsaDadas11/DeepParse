import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from service.core.chat import run_answer_pipeline
from service.core.rag.utils.es_conn import ESConnection
from service.core.retrieval_runtime import retrieve_content
from utils.database import get_session_factory


DEFAULT_CASES_PATH = Path(__file__).with_name("resume_generation_benchmark_manual_v2.json")
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

CITATION_PATTERN = re.compile(r"##(\d+)\$\$")
NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"(?P<number>[+-]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|个百分点|年|月|日|条|款|项|号|版)?"
)
ABSTAIN_MARKERS = (
    "无法回答",
    "无法根据参考资料回答",
    "未包含",
    "不包含",
    "未找到",
    "无法确定",
    "不足以回答",
    "缺少直接信息",
    "没有直接信息",
)
LEGAL_NUMBER_UNITS = {"条", "款", "项", "号", "版", "年", "月", "日"}
RATIO_UNITS = {"%", "个百分点"}
STRONG_SEMANTIC_CASE_TYPES = {
    "case_reasoning",
    "case_holding",
    "comparison_pair_text",
    "contract_clause",
    "judicial_interpretation",
    "local_vs_national",
    "paraphrase_query",
    "colloquial_query",
    "ambiguous_query",
    "procedure_requirements",
    "procedure_materials",
}
STRONG_SEMANTIC_QUESTION_HINTS = (
    "解释",
    "说明",
    "含义",
    "理由",
    "争议焦点",
    "裁判要旨",
    "比较",
    "对比",
    "区别",
    "差异",
    "总结",
    "归纳",
    "适用边界",
    "构成要件",
)


def normalize_text(value: str) -> str:
    return "".join(str(value).lower().split())


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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


def contains_expected_phrases(answer: str, case: dict) -> bool:
    normalized_answer = normalize_text(answer)
    any_phrases = [normalize_text(item) for item in (case.get("expected_any_phrases") or []) if item]
    all_phrases = [normalize_text(item) for item in (case.get("expected_all_phrases") or []) if item]

    has_any = True if not any_phrases else any(
        phrase in normalized_answer or numeric_phrase_matches(answer, phrase)
        for phrase in any_phrases
    )
    has_all = True if not all_phrases else all(
        phrase in normalized_answer or numeric_phrase_matches(answer, phrase)
        for phrase in all_phrases
    )
    return has_any and has_all


def parse_numeric_mentions(text: str) -> list[tuple[float, str]]:
    mentions: list[tuple[float, str]] = []
    for match in NUMBER_WITH_UNIT_PATTERN.finditer(str(text or "")):
        raw_number = match.group("number")
        if not raw_number:
            continue
        unit = match.group("unit") or ""
        if not unit:
            continue
        number = float(raw_number.replace(",", ""))
        if unit in LEGAL_NUMBER_UNITS:
            mentions.append((number, "legal_anchor"))
        elif unit in RATIO_UNITS:
            mentions.append((number, "ratio"))
    return mentions


def numeric_phrase_matches(answer: str, expected_phrase: str) -> bool:
    expected_mentions = parse_numeric_mentions(expected_phrase)
    answer_mentions = parse_numeric_mentions(answer)
    if not expected_mentions or not answer_mentions:
        return False

    for expected_value, expected_kind in expected_mentions:
        for answer_value, answer_kind in answer_mentions:
            if expected_kind != answer_kind:
                continue
            if expected_kind == "legal_anchor":
                if abs(expected_value - answer_value) <= 0.0:
                    return True
            else:
                scale = max(abs(expected_value), abs(answer_value), 1.0)
                if abs(expected_value - answer_value) / scale <= 0.02:
                    return True
    return False


def is_abstention(answer: str) -> bool:
    normalized_answer = normalize_text(answer)
    return any(normalize_text(marker) in normalized_answer for marker in ABSTAIN_MARKERS)


def extract_citation_count(answer: str) -> int:
    return len(set(CITATION_PATTERN.findall(answer or "")))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _is_retryable_generation_error(exc: Exception) -> bool:
    if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


def _retry_delay_seconds(attempt: int, initial_seconds: float, max_seconds: float) -> float:
    return min(max_seconds, initial_seconds * (2 ** attempt))


def answer_question(
    user_id: str,
    question: str,
    top_k: int,
    *,
    case_id: str = "",
    max_attempts: int = 5,
    retry_initial_seconds: float = 10.0,
    retry_max_seconds: float = 120.0,
) -> dict:
    references = retrieve_content(user_id, question, top_k=top_k)

    max_attempts = max(1, max_attempts)
    for attempt in range(max_attempts):
        try:
            result = run_answer_pipeline(question, references)
            return {
                "answer": result["answer"],
                "mode": result["mode"],
                "documents": result["documents"],
                "answer_audit": result.get("answer_audit", {}),
                "route_reason": result.get("route_reason"),
                "selected_mode": result.get("selected_mode"),
                "fallback_mode": result.get("fallback_mode"),
                "answer_guard": result.get("answer_guard", {}),
            }
        except Exception as exc:
            if not _is_retryable_generation_error(exc):
                raise
            if attempt == max_attempts - 1:
                raise
            backoff_seconds = _retry_delay_seconds(attempt, retry_initial_seconds, retry_max_seconds)
            label = f" case={case_id}" if case_id else ""
            print(
                f"[generation_eval] retryable error on{label} attempt {attempt + 1}/{max_attempts}: "
                f"{type(exc).__name__}; sleeping {backoff_seconds:.1f}s",
                file=sys.stderr,
            )
            time.sleep(backoff_seconds)

    raise RuntimeError("generation evaluation exhausted retry budget")


def compute_metrics(case_results: list[dict]) -> dict:
    total = len(case_results)
    latencies = [item["latency_ms"] for item in case_results]
    answerable_cases = [item for item in case_results if not item["expect_abstain"]]
    abstain_cases = [item for item in case_results if item["expect_abstain"]]

    answerable_total = len(answerable_cases)
    abstain_total = len(abstain_cases)

    accuracy_hits = sum(1 for item in answerable_cases if item["answer_hit"])
    citation_hits = sum(1 for item in answerable_cases if item["citation_hit"])
    grounded_hits = sum(1 for item in answerable_cases if item["grounded_hit"])
    abstain_hits = sum(1 for item in abstain_cases if item["abstain_hit"])

    rule_answers = sum(1 for item in case_results if item["answer_mode"] == "rule")
    model_answers = sum(1 for item in case_results if item["answer_mode"] == "model")

    return {
        "cases": total,
        "answerable_cases": answerable_total,
        "abstain_cases": abstain_total,
        "answer_accuracy": accuracy_hits / answerable_total if answerable_total else 0.0,
        "citation_support_rate": citation_hits / answerable_total if answerable_total else 0.0,
        "grounded_answer_rate": grounded_hits / answerable_total if answerable_total else 0.0,
        "abstain_success_rate": abstain_hits / abstain_total if abstain_total else 0.0,
        "hallucination_rate": 1.0 - (abstain_hits / abstain_total) if abstain_total else 0.0,
        "rule_answer_rate": rule_answers / total if total else 0.0,
        "model_answer_rate": model_answers / total if total else 0.0,
        "avg_latency_ms": sum(latencies) / total if total else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }


def summarize_metrics(case_results: list[dict]) -> dict:
    summary = compute_metrics(case_results)
    case_types = sorted({item.get("case_type", "unknown") for item in case_results})
    summary["by_case_type"] = {
        case_type: compute_metrics(
            [item for item in case_results if item.get("case_type", "unknown") == case_type]
        )
        for case_type in case_types
    }
    semantic_cases = [item for item in case_results if is_strong_semantic_case(item)]
    summary["strong_semantic_subset"] = compute_metrics(semantic_cases)
    return summary


def is_strong_semantic_case(item: dict) -> bool:
    case_type = item.get("case_type", "unknown")
    question = item.get("question", "") or ""
    return case_type in STRONG_SEMANTIC_CASE_TYPES or any(
        token in question for token in STRONG_SEMANTIC_QUESTION_HINTS
    )


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
        "CHAT_COMPLETION_TIMEOUT_SECONDS",
        "GENERATION_EVAL_MAX_ATTEMPTS",
        "GENERATION_EVAL_RETRY_INITIAL_SECONDS",
        "GENERATION_EVAL_RETRY_MAX_SECONDS",
    )
    return {key: os.getenv(key) for key in keys}


def build_case_result(case: dict, result: dict, latency_ms: float) -> dict:
    answer = result["answer"]
    citation_count = extract_citation_count(answer)
    expect_abstain = bool(case.get("expect_abstain"))
    answer_hit = False if expect_abstain else contains_expected_phrases(answer, case)
    abstain_hit = is_abstention(answer) if expect_abstain else False
    citation_hit = citation_count >= int(case.get("min_citations", 1))
    grounded_hit = answer_hit and citation_hit

    return {
        "id": case["id"],
        "case_type": case.get("case_type", "unknown"),
        "question": case["question"],
        "expect_abstain": expect_abstain,
        "answer_hit": answer_hit,
        "citation_hit": citation_hit,
        "grounded_hit": grounded_hit,
        "abstain_hit": abstain_hit,
        "answer_mode": result["mode"],
        "citation_count": citation_count,
        "latency_ms": round(latency_ms, 2),
        "answer": answer,
        "top_documents": [item.get("document_name", "") for item in result["documents"]],
        "route_reason": result.get("route_reason") or result.get("answer_audit", {}).get("route_reason"),
        "selected_mode": result.get("selected_mode") or result.get("answer_audit", {}).get("selected_mode"),
        "fallback_mode": result.get("fallback_mode") or result.get("answer_audit", {}).get("fallback_mode"),
        "answer_guard": result.get("answer_guard") or result.get("answer_audit", {}).get("answer_guard", {}),
    }


def build_error_case_result(case: dict, exc: Exception, latency_ms: float) -> dict:
    expect_abstain = bool(case.get("expect_abstain"))
    return {
        "id": case["id"],
        "case_type": case.get("case_type", "unknown"),
        "question": case["question"],
        "expect_abstain": expect_abstain,
        "answer_hit": False,
        "citation_hit": False,
        "grounded_hit": False,
        "abstain_hit": False,
        "answer_mode": "error",
        "citation_count": 0,
        "latency_ms": round(latency_ms, 2),
        "answer": "",
        "top_documents": [],
        "route_reason": "generation_eval_error",
        "selected_mode": None,
        "fallback_mode": None,
        "answer_guard": {},
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def build_report(user_id: str, case_results: list[dict], top_k: int) -> dict:
    summary = summarize_metrics(case_results)
    summary["knowledgebase_files"] = get_knowledgebase_count(user_id)
    summary["chunk_count"] = get_chunk_count(user_id)
    summary["top_k"] = top_k

    return {
        "summary": summary,
        "cases": case_results,
        "config_snapshot": runtime_config_snapshot(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def write_report_checkpoint(output_path: Path | None, report: dict) -> None:
    if not output_path:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)


def load_resume_results(output_path: Path | None) -> list[dict]:
    if not output_path or not output_path.exists():
        return []
    try:
        report = json.loads(output_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return []
    existing = report.get("cases", [])
    return existing if isinstance(existing, list) else []


def evaluate(
    user_id: str,
    cases_path: Path,
    top_k: int,
    *,
    output_path: Path | None = None,
    max_attempts: int = 5,
    retry_initial_seconds: float = 10.0,
    retry_max_seconds: float = 120.0,
    resume: bool = False,
    continue_on_case_error: bool = False,
) -> dict:
    cases = load_cases(cases_path)
    case_results: list[dict] = load_resume_results(output_path) if resume else []
    completed_ids = {item.get("id") for item in case_results}

    for case in cases:
        if case["id"] in completed_ids:
            continue

        start = time.perf_counter()
        try:
            result = answer_question(
                user_id,
                case["question"],
                top_k,
                case_id=case["id"],
                max_attempts=max_attempts,
                retry_initial_seconds=retry_initial_seconds,
                retry_max_seconds=retry_max_seconds,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            case_results.append(build_case_result(case, result, latency_ms))
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            if not continue_on_case_error:
                write_report_checkpoint(output_path, build_report(user_id, case_results, top_k))
                raise
            print(
                f"[generation_eval] case {case['id']} failed after retries: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            case_results.append(build_error_case_result(case, exc, latency_ms))

        write_report_checkpoint(
            output_path,
            build_report(user_id, case_results, top_k),
        )

    return build_report(user_id, case_results, top_k)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=_env_int("GENERATION_EVAL_MAX_ATTEMPTS", 5),
        help="Maximum outer attempts per case after retrieval. The chat layer also has its own request retry loop.",
    )
    parser.add_argument(
        "--retry-initial-seconds",
        type=float,
        default=_env_float("GENERATION_EVAL_RETRY_INITIAL_SECONDS", 10.0),
        help="Initial outer retry delay for retryable generation errors.",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=_env_float("GENERATION_EVAL_RETRY_MAX_SECONDS", 120.0),
        help="Maximum outer retry delay for retryable generation errors.",
    )
    parser.add_argument(
        "--model-timeout-seconds",
        type=int,
        default=_env_int("CHAT_COMPLETION_TIMEOUT_SECONDS", 120),
        help="Per chat completion request timeout used by the answer pipeline.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output report by skipping case ids already present in it.",
    )
    parser.add_argument(
        "--continue-on-case-error",
        action="store_true",
        help="Record exhausted case errors as failed cases instead of aborting the whole evaluation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["CHAT_COMPLETION_TIMEOUT_SECONDS"] = str(max(1, args.model_timeout_seconds))
    os.environ["GENERATION_EVAL_MAX_ATTEMPTS"] = str(max(1, args.max_attempts))
    os.environ["GENERATION_EVAL_RETRY_INITIAL_SECONDS"] = str(max(0.0, args.retry_initial_seconds))
    os.environ["GENERATION_EVAL_RETRY_MAX_SECONDS"] = str(max(0.0, args.retry_max_seconds))

    output_path = Path(args.output) if args.output else None
    report = evaluate(
        args.user_id,
        Path(args.cases),
        args.top_k,
        output_path=output_path,
        max_attempts=max(1, args.max_attempts),
        retry_initial_seconds=max(0.0, args.retry_initial_seconds),
        retry_max_seconds=max(0.0, args.retry_max_seconds),
        resume=args.resume,
        continue_on_case_error=args.continue_on_case_error,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    write_report_checkpoint(output_path, report)


if __name__ == "__main__":
    main()
