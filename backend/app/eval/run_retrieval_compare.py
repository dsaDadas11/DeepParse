import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from eval.run_retrieval_eval import (
    DEFAULT_CASES_PATH,
    dedupe_documents,
    find_document_rank,
    find_evidence_rank,
    get_chunk_count,
    get_knowledgebase_count,
    load_cases,
    summarize_metrics,
)
from service.core.rag.settings import PAGERANK_FLD
from service.core.rag.utils.doc_store_conn import FusionExpr, OrderByExpr
from service.core.rag.nlp import rag_tokenizer
from service.core.rag.nlp.search_v2 import index_name
from service.core.rag_config import (
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VECTOR_SIMILARITY_WEIGHT,
    DENSE_SIMILARITY_FALLBACK,
    FUSION_WEIGHT_TEXT,
    FUSION_WEIGHT_VECTOR,
    RERANK_CANDIDATE_CAP,
    RERANK_PAGE_LIMIT,
)
from service.core.retrieval import dealer
from service.core.retrieval_runtime import retrieve_content


GENERIC_RANK_FEATURE = {PAGERANK_FLD: 10}
BASELINE_FUSION_WEIGHTS: dict[str, tuple[float, float]] = {
    "generic_static_hybrid": (FUSION_WEIGHT_TEXT, FUSION_WEIGHT_VECTOR),
    "generic_static_hybrid_tuned": (0.20, 0.80),
    "generic_rag_baseline": (FUSION_WEIGHT_TEXT, FUSION_WEIGHT_VECTOR),
    "generic_multi_query_hybrid": (0.20, 0.80),
    "generic_multi_query_docaware": (0.20, 0.80),
}
BASELINE_DESCRIPTIONS: dict[str, str] = {
    "generic_static_hybrid": (
        "Single-query hybrid retrieval with static fusion weights "
        f"({FUSION_WEIGHT_TEXT:.2f}/{FUSION_WEIGHT_VECTOR:.2f}) and model rerank, "
        "without finance-specific retrieval_runtime planning, diversified queries, "
        "merge/routing, announcement fallback, or intent bonus."
    ),
    "generic_static_hybrid_tuned": (
        "Single-query hybrid retrieval with tuned static fusion weights (0.20/0.80) "
        "and model rerank, without finance-specific retrieval_runtime planning, "
        "diversified queries, merge/routing, announcement fallback, or intent bonus."
    ),
    "generic_rag_baseline": (
        "Legacy alias for generic_static_hybrid."
    ),
    "generic_multi_query_hybrid": (
        "Domain-agnostic multi-query hybrid retrieval with tuned static fusion weights (0.20/0.80), "
        "model rerank, generic query cleanup, and RRF merge; without finance-specific planning, "
        "routing, announcement fallback, or intent bonus."
    ),
    "generic_multi_query_docaware": (
        "Domain-agnostic multi-query hybrid retrieval with tuned static fusion weights (0.20/0.80), "
        "model rerank, generic query cleanup, RRF merge, and generic document-aware interleaving; "
        "without finance-specific planning, routing, announcement fallback, or intent bonus."
    ),
}
MULTI_QUERY_BASELINE_MODES = {"generic_multi_query_hybrid", "generic_multi_query_docaware"}
GENERIC_QUERY_CLAUSE_SPLIT = re.compile(r"[，,。；;：:\n!?！？]+")
GENERIC_LEADING_PREFIXES = (
    "请问",
    "麻烦",
    "帮我",
    "给我",
    "发我",
    "告诉我",
    "我想看",
    "我想找",
    "我想要",
    "我需要",
    "我只想看",
    "我只想",
    "我先看",
    "先给我",
    "先看",
    "想看",
    "想找",
)
GENERIC_FOCUS_FILLERS = (
    "请问",
    "麻烦",
    "帮我",
    "给我",
    "发我",
    "告诉我",
    "我想看",
    "我想找",
    "我想要",
    "我需要",
    "我只想看",
    "我只想",
    "我先看",
    "先给我",
    "先看",
    "想看",
    "想找",
    "看一下",
    "看下",
    "找一下",
    "找下",
    "有没有",
    "有吗",
    "在哪",
    "在哪里",
)
GENERIC_RRF_K = 60


def retrieve_current(user_id: str, question: str, top_k: int) -> list[dict]:
    return retrieve_content(user_id, question, top_k=top_k)


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = re.sub(r"\s+", "", value or "")
        normalized = normalized.strip("，,。；;：:!?！？")
        if len(normalized) < 4 or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _strip_generic_prefixes(question: str) -> str:
    text = re.sub(r"\s+", "", question or "")
    changed = True
    while changed:
        changed = False
        for prefix in GENERIC_LEADING_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True
    return text.strip("，,。；;：:!?！？")


def _build_generic_query_variants(question: str) -> list[str]:
    original = re.sub(r"\s+", "", question or "")
    cleaned = _strip_generic_prefixes(original)
    clauses = [segment for segment in GENERIC_QUERY_CLAUSE_SPLIT.split(cleaned) if len(segment) >= 4]

    focus = cleaned
    for filler in GENERIC_FOCUS_FILLERS:
        focus = focus.replace(filler, "")
    focus = focus.strip("，,。；;：:!?！？")

    variants = [original, cleaned]
    if clauses:
        variants.append(clauses[0])
        longest_clause = max(clauses, key=len)
        variants.append(longest_clause)
    variants.append(focus)
    return _unique_preserve_order(variants)[:4]


def _extract_chunks_from_ranks(results: dict) -> list[dict]:
    extracted = []
    for index, chunk in enumerate(results["chunks"], start=1):
        doc_name = chunk.get("docnm_kwd", "").split("/")[-1]
        extracted.append(
            {
                "id": chunk.get("chunk_id") or chunk.get("id") or index,
                "chunk_id": chunk.get("chunk_id") or chunk.get("id") or index,
                "display_rank": chunk.get("display_rank", index),
                "source_rank": chunk.get("source_rank", chunk.get("rank", index)),
                "document_id": chunk.get("doc_id", ""),
                "document_name": doc_name,
                "content_with_weight": chunk.get("content_with_weight", ""),
            }
        )
    return extracted


def retrieve_simple_hybrid_ablation(user_id: str, question: str, top_k: int) -> list[dict]:
    results = dealer.retrieval(
        question=question,
        embd_mdl=None,
        tenant_ids=user_id,
        kb_ids=None,
        vector_similarity_weight=0.6,
        page=1,
        page_size=top_k,
        use_model_rerank=False,
        use_intent_bonus=False,
    )
    return _extract_chunks_from_ranks(results)


def _rrf_score(rank: int, k: int = GENERIC_RRF_K) -> float:
    return 1.0 / (k + rank)


def _chunk_identity(chunk: dict) -> str:
    return str(chunk.get("chunk_id") or chunk.get("id") or f"{chunk.get('document_name', '')}:{chunk.get('source_rank', 999)}")


def _merge_generic_candidate_lists(candidate_lists: list[list[dict]], limit: int) -> list[dict]:
    merged: dict[str, dict] = {}
    order: list[str] = []

    for candidate_list in candidate_lists:
        for rank, chunk in enumerate(candidate_list, start=1):
            identity = _chunk_identity(chunk)
            if identity not in merged:
                item = dict(chunk)
                item["_rrf_score"] = _rrf_score(rank)
                item["_best_rank"] = int(chunk.get("source_rank", rank) or rank)
                item["_appearance_count"] = 1
                merged[identity] = item
                order.append(identity)
            else:
                item = merged[identity]
                item["_rrf_score"] += _rrf_score(rank)
                item["_appearance_count"] += 1
                item["_best_rank"] = min(item["_best_rank"], int(chunk.get("source_rank", rank) or rank))

    ordered = sorted(
        order,
        key=lambda identity: (
            -merged[identity]["_rrf_score"],
            -merged[identity]["_appearance_count"],
            merged[identity]["_best_rank"],
            identity,
        ),
    )

    results: list[dict] = []
    for display_rank, identity in enumerate(ordered[:limit], start=1):
        item = dict(merged[identity])
        item["display_rank"] = display_rank
        item["source_rank"] = item.get("_best_rank", display_rank)
        results.append(item)
    return results


def _interleave_by_document(chunks: list[dict], limit: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    doc_order: list[str] = []
    for chunk in chunks:
        document_name = str(chunk.get("document_name", ""))
        if document_name not in grouped:
            grouped[document_name] = []
            doc_order.append(document_name)
        grouped[document_name].append(chunk)

    interleaved: list[dict] = []
    round_index = 0
    while len(interleaved) < limit:
        appended = False
        for document_name in doc_order:
            group = grouped[document_name]
            if round_index >= len(group):
                continue
            item = dict(group[round_index])
            interleaved.append(item)
            appended = True
            if len(interleaved) >= limit:
                break
        if not appended:
            break
        round_index += 1

    for display_rank, chunk in enumerate(interleaved, start=1):
        chunk["display_rank"] = display_rank
    return interleaved


def _generic_single_query_search(
    user_id: str,
    question: str,
    top_k: int,
    *,
    text_weight: float,
    vector_weight: float,
) -> dealer.SearchResult:
    candidate_size = max(top_k * RERANK_PAGE_LIMIT, RERANK_CANDIDATE_CAP)
    src = [
        "docnm_kwd",
        "content_ltks",
        "kb_id",
        "img_id",
        "title_tks",
        "important_kwd",
        "position_int",
        "company_kwd",
        "report_period_kwd",
        "report_type_kwd",
        "source_kwd",
        "table_dense_int",
        "table_headers_kwd",
        "table_rows_kwd",
        "doc_id",
        "page_num_int",
        "top_int",
        "create_timestamp_flt",
        "knowledge_graph_kwd",
        "question_kwd",
        "question_tks",
        "available_int",
        "content_with_weight",
        PAGERANK_FLD,
    ]
    filters = {"available_int": 1}
    order_by = OrderByExpr()
    highlight_fields: list[str] = []
    match_text, keywords = dealer.qryr.question(question, min_match=0.3)
    match_dense = dealer.get_vector(question, None, 1024, DEFAULT_SIMILARITY_THRESHOLD)
    q_vec = match_dense.embedding_data
    src.append(f"q_{len(q_vec)}_vec")
    fusion_expr = FusionExpr(
        "weighted_sum",
        1024,
        {"weights": f"{text_weight:.2f}, {vector_weight:.2f}"},
    )
    res = dealer.dataStore.search(
        src,
        highlight_fields,
        filters,
        [match_text, match_dense, fusion_expr],
        order_by,
        0,
        candidate_size,
        [index_name(user_id)],
        None,
        rank_feature=GENERIC_RANK_FEATURE,
    )
    total = dealer.dataStore.getTotal(res)
    if total == 0:
        match_text, _ = dealer.qryr.question(question, min_match=0.1)
        match_dense.extra_options["similarity"] = DENSE_SIMILARITY_FALLBACK
        res = dealer.dataStore.search(
            src,
            highlight_fields,
            filters,
            [match_text, match_dense, fusion_expr],
            order_by,
            0,
            candidate_size,
            [index_name(user_id)],
            None,
            rank_feature=GENERIC_RANK_FEATURE,
        )
        total = dealer.dataStore.getTotal(res)

    kwds = set()
    for keyword in keywords:
        kwds.add(keyword)
        for fine_grained in rag_tokenizer.fine_grained_tokenize(keyword).split():
            if len(fine_grained) < 2 or fine_grained in kwds:
                continue
            kwds.add(fine_grained)

    return dealer.SearchResult(
        total=total,
        ids=dealer.dataStore.getChunkIds(res),
        query_vector=q_vec,
        aggregation=dealer.dataStore.getAggregation(res, "docnm_kwd"),
        highlight=dealer.dataStore.getHighlight(res, list(kwds), "content_with_weight"),
        field=dealer.dataStore.getFields(res, src),
        keywords=list(kwds),
    )


def retrieve_generic_rag_baseline(
    user_id: str,
    question: str,
    top_k: int,
    *,
    baseline_mode: str = "generic_static_hybrid",
) -> list[dict]:
    text_weight, vector_weight = BASELINE_FUSION_WEIGHTS[baseline_mode]
    sres = _generic_single_query_search(
        user_id,
        question,
        top_k,
        text_weight=text_weight,
        vector_weight=vector_weight,
    )
    if sres.total > 0:
        try:
            sim, tsim, vsim = dealer.rerank_by_model(
                None,
                sres,
                question,
                text_weight,
                vector_weight,
                rank_feature=GENERIC_RANK_FEATURE,
                use_intent_bonus=False,
            )
        except Exception:
            sim, tsim, vsim = dealer.rerank(
                sres,
                question,
                text_weight,
                vector_weight,
                rank_feature=GENERIC_RANK_FEATURE,
                use_intent_bonus=False,
            )
    else:
        sim, tsim, vsim = [], [], []

    ranks = {"total": sres.total, "chunks": [], "doc_aggs": {}}
    has_scores = len(sim) > 0
    if has_scores:
        sim = np.asarray(sim)
        ranked_indexes = np.argsort(sim * -1)[:top_k]
    else:
        ranked_indexes = list(range(min(len(sres.ids), top_k)))

    dim = len(sres.query_vector or [])
    vector_column = f"q_{dim}_vec" if dim else ""
    zero_vector = [0.0] * dim if dim else []
    for display_rank, raw_index in enumerate(ranked_indexes, start=1):
        if has_scores and sim[raw_index] < DEFAULT_SIMILARITY_THRESHOLD:
            break
        chunk_id = sres.ids[int(raw_index)]
        chunk = sres.field[chunk_id]
        ranks["chunks"].append(
            {
                "chunk_id": chunk_id,
                "id": chunk_id,
                "display_rank": display_rank,
                "rank": int(raw_index) + 1,
                "source_rank": int(raw_index) + 1,
                "content_with_weight": chunk.get("content_with_weight", ""),
                "doc_id": chunk.get("doc_id", ""),
                "docnm_kwd": chunk.get("docnm_kwd", ""),
                "similarity": float(sim[raw_index]) if has_scores else 0.0,
                "vector_similarity": float(vsim[raw_index]) if len(vsim) > 0 else 0.0,
                "term_similarity": float(tsim[raw_index]) if len(tsim) > 0 else 0.0,
                "vector": chunk.get(vector_column, zero_vector) if vector_column else [],
            }
        )
    return _extract_chunks_from_ranks(ranks)


def retrieve_generic_multi_query_baseline(user_id: str, question: str, top_k: int, *, baseline_mode: str) -> list[dict]:
    text_weight, vector_weight = BASELINE_FUSION_WEIGHTS[baseline_mode]
    candidate_limit = max(top_k * 4, 16)
    query_variants = _build_generic_query_variants(question)
    candidate_lists = [
        retrieve_generic_rag_baseline(
            user_id,
            query_variant,
            candidate_limit,
            baseline_mode="generic_static_hybrid_tuned",
        )
        for query_variant in query_variants
    ]

    merged = _merge_generic_candidate_lists(candidate_lists, limit=max(candidate_limit, top_k * 3))
    if baseline_mode == "generic_multi_query_docaware":
        merged = _interleave_by_document(merged, limit=top_k)
    else:
        merged = merged[:top_k]

    for item in merged:
        item["term_similarity_weight"] = text_weight
        item["vector_similarity_weight"] = vector_weight
    return merged


def evaluate_mode(user_id: str, cases: list[dict], top_k: int, mode: str) -> dict:
    case_results: list[dict] = []

    for case in cases:
        question = case["question"]
        start = time.perf_counter()
        if mode == "current":
            results = retrieve_current(user_id, question, top_k)
        elif mode in MULTI_QUERY_BASELINE_MODES:
            results = retrieve_generic_multi_query_baseline(user_id, question, top_k, baseline_mode=mode)
        elif mode in BASELINE_FUSION_WEIGHTS:
            results = retrieve_generic_rag_baseline(user_id, question, top_k, baseline_mode=mode)
        elif mode == "simple_hybrid_ablation":
            results = retrieve_simple_hybrid_ablation(user_id, question, top_k)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

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
    }


def compute_delta(current: dict, baseline: dict) -> dict:
    keys = (
        "hit_at_1",
        "hit_at_3",
        "hit_at_k",
        "mrr",
        "evidence_hit_at_1",
        "evidence_hit_at_3",
        "evidence_hit_at_k",
    )
    return {key: current.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}


def compute_case_type_delta(current_report: dict, baseline_report: dict) -> dict:
    return {
        case_type: compute_delta(
            current_report["summary"]["by_case_type"].get(case_type, {}),
            baseline_report["summary"]["by_case_type"].get(case_type, {}),
        )
        for case_type in sorted(current_report["summary"]["by_case_type"].keys())
    }


def compare(user_id: str, cases_path: Path, top_k: int, baseline_mode: str) -> dict:
    cases = load_cases(cases_path)
    current = evaluate_mode(user_id, cases, top_k, mode="current")
    baseline = evaluate_mode(user_id, cases, top_k, mode=baseline_mode)
    ablation = evaluate_mode(user_id, cases, top_k, mode="simple_hybrid_ablation")

    return {
        "cases_path": str(cases_path),
        "current_label": "full_runtime_pipeline",
        "baseline_label": baseline_mode,
        "baseline_description": BASELINE_DESCRIPTIONS[baseline_mode],
        "current": current,
        "baseline": baseline,
        "generic_rag_baseline": baseline if baseline_mode == "generic_rag_baseline" else None,
        "generic_static_hybrid": baseline if baseline_mode == "generic_static_hybrid" else None,
        "generic_static_hybrid_tuned": baseline if baseline_mode == "generic_static_hybrid_tuned" else None,
        "simple_hybrid_ablation": ablation,
        "weak_baseline": ablation,
        "delta": {
            "summary": compute_delta(current["summary"], baseline["summary"]),
            "by_case_type": compute_case_type_delta(current, baseline),
        },
        "auxiliary_delta": {
            "label": "full_runtime_pipeline_vs_simple_hybrid_ablation",
            "summary": compute_delta(current["summary"], ablation["summary"]),
            "by_case_type": compute_case_type_delta(current, ablation),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--baseline-mode",
        default="generic_static_hybrid",
        choices=(
            "generic_static_hybrid",
            "generic_static_hybrid_tuned",
            "generic_multi_query_hybrid",
            "generic_multi_query_docaware",
        ),
    )
    parser.add_argument("--output")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    report = compare(args.user_id, Path(args.cases), args.top_k, args.baseline_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
