from __future__ import annotations

import argparse
import json
from pathlib import Path

from service.core.rag.utils.es_conn import ESConnection


def normalize_name(name: str) -> str:
    return (name or "").replace(" ", "").replace("（", "(").replace("）", ")").lower()


def load_expected_documents(cases_path: Path) -> list[str]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    docs: list[str] = []
    for case in cases:
        docs.extend(case.get("expected_documents", []) or [])
    # unique keep order
    seen = set()
    ordered: list[str] = []
    for doc in docs:
        if doc in seen:
            continue
        seen.add(doc)
        ordered.append(doc)
    return ordered


def fetch_index_doc_names(user_id: str, page_size: int = 2000) -> list[str]:
    es = ESConnection().es
    if not es.indices.exists(index=user_id):
        return []

    body = {
        "query": {"term": {"available_int": 1}},
        "_source": ["docnm_kwd"],
        "size": page_size,
        "sort": [{"_doc": {"order": "asc"}}],
    }
    response = es.search(index=user_id, body=body, timeout="120s", track_total_hits=False)
    names: list[str] = []
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source", {}) or {}
        raw = str(source.get("docnm_kwd", "") or "")
        if not raw:
            continue
        names.append(raw.split("/")[-1])
    return list(dict.fromkeys(names))


def audit(user_id: str, retrieval_cases: Path, generation_cases: Path) -> dict:
    expected = load_expected_documents(retrieval_cases) + load_expected_documents(generation_cases)
    expected = list(dict.fromkeys(expected))
    indexed = fetch_index_doc_names(user_id)

    norm_to_indexed = {normalize_name(name): name for name in indexed}

    found: list[dict] = []
    missing: list[str] = []
    for doc in expected:
        norm = normalize_name(doc)
        matched = norm_to_indexed.get(norm)
        if matched:
            found.append({"expected": doc, "matched": matched})
        else:
            missing.append(doc)

    return {
        "user_id": user_id,
        "expected_count": len(expected),
        "indexed_unique_count": len(indexed),
        "found_count": len(found),
        "missing_count": len(missing),
        "found": found,
        "missing": missing,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--retrieval-cases", required=True)
    parser.add_argument("--generation-cases", required=True)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit(
        args.user_id,
        Path(args.retrieval_cases),
        Path(args.generation_cases),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
