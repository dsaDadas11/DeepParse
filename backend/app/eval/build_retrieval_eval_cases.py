import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from runtime_config import ALLOWED_UPLOAD_EXTENSIONS
from service.core.document_metadata import parse_document_metadata

DEFAULT_OUTPUT_PATH = Path(__file__).with_name("retrieval_eval_cases.json")

BUCKETS = [
    "statute_exact",
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
]


def gather_files(source_dir: Path) -> list[Path]:
    return sorted(
        file_path
        for file_path in source_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS
    )


def _q(meta, template: str) -> str:
    subject = meta.company or meta.base_name
    return template.format(subject=subject)


def build_cases(source_dir: Path, evidence_limit: int) -> list[dict]:
    files = gather_files(source_dir)
    metas = [parse_document_metadata(file.name) for file in files]

    cases: list[dict] = []
    idx = 1

    for meta in metas:
        subject = meta.company or meta.base_name
        expected = [meta.file_name]

        if meta.doc_type in {"法律", "条例", "地方性法规"}:
            cases.append({"id": f"legal_{idx:03d}", "case_type": "statute_exact", "question": f"{subject}原文", "expected_documents": expected})
            idx += 1
            cases.append({"id": f"legal_{idx:03d}", "case_type": "article_exact", "question": f"{subject}第几条怎么规定", "expected_documents": expected})
            idx += 1
            cases.append({"id": f"legal_{idx:03d}", "case_type": "version_conflict", "question": f"{subject}现行版本和修订版有什么区别", "expected_documents": expected})
            idx += 1

        if meta.doc_type == "办事指南":
            cases.append({"id": f"legal_{idx:03d}", "case_type": "procedure_requirements", "question": f"{subject}申请条件是什么", "expected_documents": expected})
            idx += 1
            cases.append({"id": f"legal_{idx:03d}", "case_type": "procedure_materials", "question": f"{subject}申请材料有哪些", "expected_documents": expected})
            idx += 1
            cases.append({"id": f"legal_{idx:03d}", "case_type": "colloquial_query", "question": f"{subject}要准备啥材料多久能办完", "expected_documents": expected})
            idx += 1

        if meta.doc_type == "合同范本":
            cases.append({"id": f"legal_{idx:03d}", "case_type": "contract_clause", "question": f"{subject}违约责任和解除条件怎么写", "expected_documents": expected})
            idx += 1

        if meta.doc_type in {"指导性案例", "裁判文书"}:
            cases.append({"id": f"legal_{idx:03d}", "case_type": "case_holding", "question": f"{subject}裁判要旨是什么", "expected_documents": expected})
            idx += 1
            cases.append({"id": f"legal_{idx:03d}", "case_type": "case_reasoning", "question": f"{subject}裁判理由和争点怎么认定", "expected_documents": expected})
            idx += 1

        if meta.doc_type == "司法解释":
            cases.append({"id": f"legal_{idx:03d}", "case_type": "judicial_interpretation", "question": f"{subject}司法解释怎么规定", "expected_documents": expected})
            idx += 1

        if meta.doc_type in {"地方性法规", "条例"}:
            cases.append({"id": f"legal_{idx:03d}", "case_type": "local_vs_national", "question": f"{subject}和国家层面规定是否一致", "expected_documents": expected})
            idx += 1

        # 泛化问法
        cases.append({"id": f"legal_{idx:03d}", "case_type": "paraphrase_query", "question": f"请帮我定位{subject}相关规定并总结适用边界", "expected_documents": expected})
        idx += 1
        cases.append({"id": f"legal_{idx:03d}", "case_type": "ambiguous_query", "question": f"{subject}这个到底怎么处理", "expected_documents": expected})
        idx += 1

    # abstain bucket
    abstains = [
        "不存在的法律第999条怎么规定",
        "某未知地区2022年特别条例是否有效",
        "虚构指导性案例999号裁判要旨是什么",
        "完全没在语料里的许可申请需要什么材料",
    ]
    for q in abstains:
        cases.append(
            {
                "id": f"legal_{idx:03d}",
                "case_type": "abstain_required",
                "question": q,
                "expected_documents": [],
            }
        )
        idx += 1

    # 限制体量，保持可跑
    return cases[: max(80, evidence_limit * 2)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--evidence-limit", type=int, default=40)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cases = build_cases(Path(args.source_dir), args.evidence_limit)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generated] cases={len(cases)} output={output_path}")
    print(f"[buckets] {', '.join(BUCKETS)}")


if __name__ == "__main__":
    main()
