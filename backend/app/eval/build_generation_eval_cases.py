import argparse
import json
from pathlib import Path

DEFAULT_RETRIEVAL_CASES_PATH = Path(__file__).with_name("retrieval_eval_cases.json")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("generation_eval_cases.json")
DEFAULT_SEMANTIC_OUTPUT_PATH = Path(__file__).with_name("semantic_generation_regression.json")

STRONG_SEMANTIC_CASE_TYPES = {
    "procedure_requirements",
    "procedure_materials",
    "contract_clause",
    "case_holding",
    "case_reasoning",
    "judicial_interpretation",
    "local_vs_national",
    "paraphrase_query",
    "colloquial_query",
    "ambiguous_query",
    "version_conflict",
}


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_strong_semantic_case(case: dict) -> bool:
    return case.get("case_type") in STRONG_SEMANTIC_CASE_TYPES


def build_cases(retrieval_cases_path: Path) -> list[dict]:
    retrieval_cases = load_cases(retrieval_cases_path)
    generation_cases: list[dict] = []

    supported = {
        "article_exact",
        "procedure_requirements",
        "procedure_materials",
        "contract_clause",
        "case_holding",
        "case_reasoning",
        "judicial_interpretation",
        "version_conflict",
        "local_vs_national",
        "paraphrase_query",
        "colloquial_query",
        "ambiguous_query",
        "abstain_required",
    }

    for index, case in enumerate(retrieval_cases, start=1):
        if case.get("case_type") not in supported:
            continue
        generation_cases.append(
            {
                "id": f"gen_{index:03d}",
                "case_type": case["case_type"],
                "question": case["question"],
                "expected_documents": case.get("expected_documents", []),
                "min_citations": 1,
                "expect_abstain": case.get("case_type") == "abstain_required",
            }
        )

    return generation_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-cases", default=str(DEFAULT_RETRIEVAL_CASES_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--semantic-output", default=str(DEFAULT_SEMANTIC_OUTPUT_PATH))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cases = build_cases(Path(args.retrieval_cases))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    semantic_cases = [case for case in cases if is_strong_semantic_case(case)]
    semantic_output_path = Path(args.semantic_output)
    semantic_output_path.write_text(json.dumps(semantic_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[generated] generation_cases={len(cases)} output={output_path} "
        f"semantic_cases={len(semantic_cases)} semantic_output={semantic_output_path}"
    )


if __name__ == "__main__":
    main()
