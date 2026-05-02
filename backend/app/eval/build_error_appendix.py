import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


DEFAULT_RETRIEVAL_REPORT = Path(__file__).with_name("latest_retrieval_compare_report.json")
DEFAULT_GENERATION_REPORT = Path(__file__).with_name("latest_generation_eval_report.json")
DEFAULT_RETRIEVAL_CASES = Path(__file__).with_name("retrieval_eval_cases.json")
DEFAULT_GENERATION_CASES = Path(__file__).with_name("generation_eval_cases.json")
DEFAULT_MANIFEST = Path(__file__).with_name("benchmark_manifest.json")
DEFAULT_OUTPUT = Path(__file__).with_name("latest_error_appendix.json")
DEFAULT_MARKDOWN_OUTPUT = Path(__file__).with_name("latest_error_appendix.md")

NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"(?P<number>[+-]?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>个百分点|百万元|千万元|亿元|万元|元|%|亿|万|倍)?"
)
RAW_NUMBER_PATTERN = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")
YEAR_PATTERN = re.compile(r"20\d{2}")
BROKER_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,8}证券")

MONEY_UNIT_SCALE = {
    "元": 1.0,
    "千元": 1_000.0,
    "万元": 10_000.0,
    "百万元": 1_000_000.0,
    "千万元": 10_000_000.0,
    "亿": 100_000_000.0,
    "亿元": 100_000_000.0,
}
RATIO_UNITS = {"%", "个百分点"}
ABSTAIN_MARKERS = (
    "未在当前检索结果中找到",
    "参考资料中没有直接提供",
    "没有直接披露",
    "暂未披露",
    "无法根据当前资料",
    "无法从当前资料",
    "缺少直接依据",
    "没有足够依据",
)
NUMERIC_GROWTH_HINTS = ("同比", "环比", "增长率", "增速", "pct")
SCOPE_HINTS = ("子公司", "分部", "分产品", "分地区", "单季", "单季度", "q1", "q2", "q3", "q4")
GENERIC_RISK_HINTS = (
    "全面风险管理",
    "风险管控",
    "风险合规",
    "信用风险",
    "市场风险",
    "流动性风险",
    "操作风险",
    "声誉风险",
)
PERIOD_HINTS = ("年报", "中报", "半年报", "三季报", "一季报", "h1", "q1", "q2", "q3", "q4", "前三季度")
NUMERIC_HINTS = ("营收", "收入", "归母净利润", "净利润", "eps", "roe", "分红", "派息", "利润")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(value: str) -> str:
    return "".join(str(value or "").lower().split())


def shorten(value: str, limit: int = 160) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def parse_numeric_mentions(text: str) -> list[tuple[float, str]]:
    mentions: list[tuple[float, str]] = []
    for match in NUMBER_WITH_UNIT_PATTERN.finditer(str(text or "")):
        raw_number = match.group("number")
        unit = match.group("unit") or ""
        if not raw_number or not unit:
            continue
        value = float(raw_number.replace(",", ""))
        if unit in MONEY_UNIT_SCALE:
            mentions.append((value * MONEY_UNIT_SCALE[unit], "money"))
        elif unit in RATIO_UNITS:
            mentions.append((value, "ratio"))
    return mentions


def raw_number_count(text: str) -> int:
    return len(RAW_NUMBER_PATTERN.findall(str(text or "")))


def is_abstention(answer: str) -> bool:
    normalized = normalize_text(answer)
    return any(normalize_text(marker) in normalized for marker in ABSTAIN_MARKERS)


def normalize_doc_list(values: list[str] | None) -> list[str]:
    return [normalize_text(value) for value in (values or []) if value]


def top_docs_hit(top_documents: list[str], expected_documents: list[str]) -> bool:
    expected = set(normalize_doc_list(expected_documents))
    if not expected:
        return True
    return any(normalize_text(doc) in expected for doc in (top_documents or []))


def extract_years(text: str) -> list[str]:
    return sorted(set(YEAR_PATTERN.findall(str(text or ""))))


def extract_brokers(text: str) -> list[str]:
    return sorted(set(BROKER_PATTERN.findall(str(text or ""))))


def has_time_conflict(question: str, answer: str, top_documents: list[str]) -> bool:
    question_years = extract_years(question)
    if not question_years:
        return False
    combined_years = extract_years(" ".join([answer or "", *top_documents]))
    if not combined_years:
        return False
    return not any(year in combined_years for year in question_years)


def has_broker_conflict(question: str, top_documents: list[str]) -> bool:
    brokers = extract_brokers(question)
    if not brokers:
        return False
    combined = normalize_text(" ".join(top_documents))
    return not any(normalize_text(broker) in combined for broker in brokers)


def query_traits(question: str) -> list[str]:
    normalized = normalize_text(question)
    traits: list[str] = []
    if extract_years(question) or any(token in normalized for token in PERIOD_HINTS):
        traits.append("period_constrained")
    if "证券" in question or "研报" in question or "点评" in question:
        traits.append("broker_or_commentary")
    if "风险" in question:
        traits.append("risk")
    if any(token in question for token in NUMERIC_HINTS):
        traits.append("numeric")
    if any(token in question for token in ("公告", "摘要", "表格")):
        traits.append("doc_type")
    if len(traits) >= 3:
        traits.append("multi_constraint")
    return traits or ["generic"]


def classify_numeric_error(result: dict, case: dict) -> str:
    answer = result.get("answer", "")
    top_documents = result.get("top_documents", [])
    expected_documents = case.get("expected_documents", [])
    expected_phrases = [*(case.get("expected_any_phrases") or []), *(case.get("expected_all_phrases") or [])]

    if is_abstention(answer):
        return "false_refusal"
    if not top_docs_hit(top_documents, expected_documents):
        return "retrieval_miss"
    if has_time_conflict(case.get("question", ""), answer, top_documents):
        return "time_anchor_conflict"
    if any(token in normalize_text(answer) for token in NUMERIC_GROWTH_HINTS) and not any(
        token in normalize_text(case.get("question", "")) for token in NUMERIC_GROWTH_HINTS
    ):
        return "growth_vs_actual_mixup"

    answer_mentions = parse_numeric_mentions(answer)
    expected_mentions = parse_numeric_mentions(" ".join(expected_phrases))
    if not answer_mentions:
        if raw_number_count(answer) > 0:
            return "raw_value_without_unit"
        return "missing_numeric_value"

    for expected_value, expected_kind in expected_mentions:
        for answer_value, answer_kind in answer_mentions:
            if expected_kind != answer_kind:
                continue
            if answer_value == 0 or expected_value == 0:
                continue
            ratio = max(answer_value, expected_value) / max(1.0, min(answer_value, expected_value))
            if ratio >= 80:
                return "unit_scale_mismatch"

    if any(token in normalize_text(answer) for token in SCOPE_HINTS):
        return "partial_scope_or_subsidiary_value"
    return "other_numeric_mismatch"


def classify_risk_error(result: dict, case: dict) -> str:
    answer = result.get("answer", "")
    top_documents = result.get("top_documents", [])
    expected_documents = case.get("expected_documents", [])
    normalized_answer = normalize_text(answer)

    if is_abstention(answer):
        return "false_refusal"
    if not top_docs_hit(top_documents, expected_documents):
        return "retrieval_miss"
    if has_time_conflict(case.get("question", ""), answer, top_documents) or has_broker_conflict(
        case.get("question", ""), top_documents
    ):
        return "constraint_conflict"
    if any(normalize_text(token) in normalized_answer for token in GENERIC_RISK_HINTS):
        return "generic_risk_governance"
    return "missing_key_risk_points"


def summarize_labels(examples: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in examples:
        label = item["label"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])))


def limited_examples(examples: list[dict], limit: int = 5) -> list[dict]:
    return examples[:limit]


def build_numeric_errors(generation_report: dict, generation_cases: list[dict]) -> dict:
    case_map = {case["id"]: case for case in generation_cases}
    examples: list[dict] = []
    for result in generation_report.get("cases", []):
        if result.get("case_type") != "numeric_evidence" or result.get("answer_hit"):
            continue
        case = case_map.get(result["id"], {})
        examples.append(
            {
                "id": result["id"],
                "label": classify_numeric_error(result, case),
                "question": result.get("question", ""),
                "expected_documents": case.get("expected_documents", []),
                "expected_any_phrases": case.get("expected_any_phrases", []),
                "top_documents": result.get("top_documents", []),
                "answer": result.get("answer", ""),
            }
        )

    return {
        "count": len(examples),
        "by_label": summarize_labels(examples),
        "examples": limited_examples(examples, limit=8),
    }


def build_risk_errors(generation_report: dict, generation_cases: list[dict]) -> dict:
    case_map = {case["id"]: case for case in generation_cases}
    examples: list[dict] = []
    for result in generation_report.get("cases", []):
        if result.get("case_type") != "risk_evidence" or result.get("answer_hit"):
            continue
        case = case_map.get(result["id"], {})
        examples.append(
            {
                "id": result["id"],
                "label": classify_risk_error(result, case),
                "question": result.get("question", ""),
                "expected_documents": case.get("expected_documents", []),
                "expected_any_phrases": case.get("expected_any_phrases", []),
                "top_documents": result.get("top_documents", []),
                "answer": result.get("answer", ""),
            }
        )

    return {
        "count": len(examples),
        "by_label": summarize_labels(examples),
        "examples": limited_examples(examples, limit=8),
    }


def build_evidence_misses(retrieval_report: dict) -> dict:
    current_cases = retrieval_report.get("current", {}).get("cases", [])
    by_case_type: dict[str, dict[str, int]] = {}
    by_trait: dict[str, dict[str, int]] = {}
    examples: list[dict] = []

    for result in current_cases:
        if not result.get("evidence_expected"):
            continue
        evidence_rank = result.get("evidence_rank")
        miss_top1 = evidence_rank != 1
        miss_top3 = evidence_rank is None or evidence_rank > 3
        if not miss_top1 and not miss_top3:
            continue

        case_type = result.get("case_type", "unknown")
        case_bucket = by_case_type.setdefault(case_type, {"evidence_cases": 0, "miss_top1": 0, "miss_top3": 0})
        case_bucket["evidence_cases"] += 1
        case_bucket["miss_top1"] += 1 if miss_top1 else 0
        case_bucket["miss_top3"] += 1 if miss_top3 else 0

        traits = query_traits(result.get("question", ""))
        for trait in traits:
            trait_bucket = by_trait.setdefault(trait, {"miss_top1": 0, "miss_top3": 0})
            trait_bucket["miss_top1"] += 1 if miss_top1 else 0
            trait_bucket["miss_top3"] += 1 if miss_top3 else 0

        if miss_top3:
            examples.append(
                {
                    "id": result.get("id"),
                    "case_type": case_type,
                    "question": result.get("question", ""),
                    "evidence_rank": evidence_rank,
                    "query_traits": traits,
                    "top_documents": result.get("top_documents", []),
                }
            )

    return {
        "by_case_type": dict(sorted(by_case_type.items())),
        "by_query_trait": dict(sorted(by_trait.items())),
        "top3_miss_examples": limited_examples(examples, limit=10),
    }


def build_abstention_errors(generation_report: dict) -> dict:
    false_refusals: list[dict] = []
    missed_refusals: list[dict] = []

    for result in generation_report.get("cases", []):
        if result.get("expect_abstain"):
            if not result.get("abstain_hit"):
                missed_refusals.append(
                    {
                        "id": result.get("id"),
                        "case_type": result.get("case_type", "unknown"),
                        "question": result.get("question", ""),
                        "answer": result.get("answer", ""),
                        "top_documents": result.get("top_documents", []),
                    }
                )
            continue

        if is_abstention(result.get("answer", "")):
            false_refusals.append(
                {
                    "id": result.get("id"),
                    "case_type": result.get("case_type", "unknown"),
                    "question": result.get("question", ""),
                    "answer": result.get("answer", ""),
                    "top_documents": result.get("top_documents", []),
                }
            )

    return {
        "false_refusal_count": len(false_refusals),
        "missed_refusal_count": len(missed_refusals),
        "false_refusal_examples": limited_examples(false_refusals),
        "missed_refusal_examples": limited_examples(missed_refusals),
    }


def build_markdown(appendix: dict) -> str:
    lines = [
        "# Error Analysis Appendix",
        "",
        f"- benchmark_tag: `{appendix.get('benchmark_tag', 'unknown')}`",
        f"- retrieval_cases: `{appendix['retrieval_summary'].get('cases', 0)}`",
        f"- generation_cases: `{appendix['generation_summary'].get('cases', 0)}`",
        "",
        "## Numeric Errors",
        "",
    ]

    numeric = appendix["numeric_answer_errors"]
    if numeric["by_label"]:
        for label, count in numeric["by_label"].items():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Risk Evidence Errors", ""])
    risk = appendix["risk_evidence_errors"]
    if risk["by_label"]:
        for label, count in risk["by_label"].items():
            lines.append(f"- {label}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Evidence Miss Traits", ""])
    traits = appendix["evidence_misses"]["by_query_trait"]
    if traits:
        for trait, counts in traits.items():
            lines.append(
                f"- {trait}: top1_miss={counts.get('miss_top1', 0)}, top3_miss={counts.get('miss_top3', 0)}"
            )
    else:
        lines.append("- none")

    lines.extend(["", "## Abstention Errors", ""])
    abstention = appendix["abstention_errors"]
    lines.append(f"- false_refusal: {abstention['false_refusal_count']}")
    lines.append(f"- missed_refusal: {abstention['missed_refusal_count']}")

    if numeric["examples"]:
        lines.extend(["", "## Numeric Examples", ""])
        for item in numeric["examples"][:5]:
            lines.append(f"- [{item['label']}] {shorten(item['question'])}")
            lines.append(f"  answer: {shorten(item['answer'])}")
            lines.append(f"  top_docs: {', '.join(item.get('top_documents', [])[:3])}")

    if risk["examples"]:
        lines.extend(["", "## Risk Examples", ""])
        for item in risk["examples"][:5]:
            lines.append(f"- [{item['label']}] {shorten(item['question'])}")
            lines.append(f"  answer: {shorten(item['answer'])}")
            lines.append(f"  top_docs: {', '.join(item.get('top_documents', [])[:3])}")

    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-report", default=str(DEFAULT_RETRIEVAL_REPORT))
    parser.add_argument("--generation-report", default=str(DEFAULT_GENERATION_REPORT))
    parser.add_argument("--retrieval-cases", default=str(DEFAULT_RETRIEVAL_CASES))
    parser.add_argument("--generation-cases", default=str(DEFAULT_GENERATION_CASES))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--markdown-output", default=str(DEFAULT_MARKDOWN_OUTPUT))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    retrieval_report = load_json(Path(args.retrieval_report))
    generation_report = load_json(Path(args.generation_report))
    retrieval_cases = load_json(Path(args.retrieval_cases))
    generation_cases = load_json(Path(args.generation_cases))
    manifest = load_json(Path(args.manifest))

    appendix = {
        "benchmark_tag": (manifest or {}).get("benchmark_tag"),
        "benchmark_fingerprint": (manifest or {}).get("benchmark_fingerprint"),
        "retrieval_summary": (
            retrieval_report.get("current", {}).get("summary", {}) if isinstance(retrieval_report, dict) else {}
        ),
        "generation_summary": generation_report.get("summary", {}) if isinstance(generation_report, dict) else {},
        "numeric_answer_errors": build_numeric_errors(generation_report, generation_cases),
        "risk_evidence_errors": build_risk_errors(generation_report, generation_cases),
        "evidence_misses": build_evidence_misses(retrieval_report),
        "abstention_errors": build_abstention_errors(generation_report),
        "retrieval_case_count": len(retrieval_cases or []),
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(appendix, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = Path(args.markdown_output)
    markdown_path.write_text(build_markdown(appendix), encoding="utf-8")
    print(json.dumps(appendix, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
