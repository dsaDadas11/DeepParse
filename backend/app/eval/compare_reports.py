import argparse
import json
from pathlib import Path


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_get(report: dict, metric: str) -> float:
    return float(report.get("summary", {}).get(metric, 0.0) or 0.0)


def diff_entry(base: float, cand: float) -> dict:
    delta = cand - base
    rel = (delta / base) if base else 0.0
    return {
        "baseline": base,
        "candidate": cand,
        "delta": delta,
        "relative_delta": rel,
    }


def classify_generation_failure(case: dict) -> str:
    expect_abstain = bool(case.get("expect_abstain"))
    answer_hit = bool(case.get("answer_hit"))
    citation_hit = bool(case.get("citation_hit"))
    abstain_hit = bool(case.get("abstain_hit"))

    if expect_abstain and not abstain_hit:
        return "拒答策略错误"
    if not expect_abstain and not answer_hit and citation_hit:
        return "生成错误"
    if not expect_abstain and answer_hit and not citation_hit:
        return "证据错位"
    if not expect_abstain and not answer_hit and not citation_hit:
        return "检索错误"
    return "通过"


def build_failure_taxonomy(report: dict) -> dict:
    taxonomy = {
        "检索错误": 0,
        "证据错位": 0,
        "生成错误": 0,
        "拒答策略错误": 0,
        "通过": 0,
    }
    for case in report.get("cases", []):
        taxonomy[classify_generation_failure(case)] += 1
    return taxonomy


def compare_reports(baseline: dict, candidate: dict) -> dict:
    retrieval_metrics = ("hit_at_1", "hit_at_3", "mrr", "evidence_hit_at_1", "evidence_hit_at_3")
    generation_metrics = (
        "answer_accuracy",
        "grounded_answer_rate",
        "citation_support_rate",
        "abstain_success_rate",
        "hallucination_rate",
    )

    compared: dict[str, dict] = {
        "retrieval": {},
        "generation": {},
    }

    for metric in retrieval_metrics:
        compared["retrieval"][metric] = diff_entry(metric_get(baseline, metric), metric_get(candidate, metric))

    for metric in generation_metrics:
        compared["generation"][metric] = diff_entry(metric_get(baseline, metric), metric_get(candidate, metric))

    compared["candidate_failure_taxonomy"] = build_failure_taxonomy(candidate)
    compared["baseline_failure_taxonomy"] = build_failure_taxonomy(baseline)
    return compared


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    baseline = load_report(Path(args.baseline))
    candidate = load_report(Path(args.candidate))
    report = compare_reports(baseline, candidate)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
