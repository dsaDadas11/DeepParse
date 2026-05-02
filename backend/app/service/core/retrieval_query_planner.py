from __future__ import annotations

import re
from typing import Any

from service.core.retrieval_intent import QueryIntentProfile, classify_query_intent


def actual_metric_terms(query: str) -> list[str]:
    text = query or ""
    metrics: list[str] = []
    if any(token in text for token in ("条", "款", "项")):
        metrics.append("法条")
    if any(token in text for token in ("申请条件", "申请材料", "办理流程", "办理时限")):
        metrics.append("程序要素")
    if any(token in text for token in ("违约责任", "解除条件", "争议解决")):
        metrics.append("合同条款")
    return unique_preserve_order(metrics)


def infer_period_scopes(query: str) -> list[str]:
    return unique_preserve_order(DATE_PATTERN.findall(query or ""))


def is_mixed_actual_forecast_query(query: str) -> bool:
    text = query or ""
    return any(token in text for token in ("现行", "修订", "生效")) and any(token in text for token in ("比较", "对比", "区别"))

ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千0-9]+(?:编|章|节|条|款|项)")
CASE_PATTERN = re.compile(r"指导性案例\s*第?\d+号|\(?\d{4}\)?[\u4e00-\u9fa5]{1,6}\d{2,8}号")
DATE_PATTERN = re.compile(r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?")
VERSION_TOKEN_PATTERN = re.compile(r"\b\d{8}\b|\d{4}[_-]\d\b|\d{4}版")
ISSUE_PATTERN = re.compile(r"第\s*([0-90-9一二三四五六七八九十百]+)\s*期")
YEAR_PATTERN = re.compile(r"((?:19|20)[0-90-9]{2})\s*年")


class RetrievalConstraints(dict):
    pass

DOC_TYPE_ALIAS: dict[str, tuple[str, ...]] = {
    "法律": ("法律", "法典", "法条"),
    "条例": ("条例", "实施条例", "办法", "规定"),
    "地方性法规": ("地方性法规", "地方条例", "地方规定"),
    "司法解释": ("司法解释", "批复", "解释"),
    "指导性案例": ("指导性案例", "典型案例"),
    "裁判文书": ("裁判文书", "判决书", "裁定书"),
    "合同范本": ("合同范本", "示范文本", "合同模板"),
    "办事指南": ("办事指南", "申请指南", "办事流程"),
    "公报": ("公报", "通告", "法规汇编"),
}

COMMON_LEGAL_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("民法典", "中华人民共和国民法典"),
    ("刑法", "中华人民共和国刑法"),
    ("行政处罚法", "中华人民共和国行政处罚法"),
    ("著作权法", "中华人民共和国著作权法"),
    ("专利法", "中华人民共和国专利法"),
    ("劳动合同", "劳动合同法"),
)


def unique_preserve_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _expand_common_legal_aliases(query: str) -> list[str]:
    expanded = [query]
    for short, full in COMMON_LEGAL_EXPANSIONS:
        if short in query and full not in query:
            expanded.append(query.replace(short, full))
    return unique_preserve_order(expanded)


def _extract_doc_type_terms(query: str, profile: QueryIntentProfile) -> list[str]:
    terms: list[str] = []
    for doc_type in profile.doc_type_terms:
        terms.extend(DOC_TYPE_ALIAS.get(doc_type, (doc_type,)))
    return unique_preserve_order(terms)


def _normalize_digits(text: str) -> str:
    return (text or "").translate(str.maketrans("0123456789", "0123456789"))


def _extract_gazette_tokens(query: str) -> tuple[list[str], list[str]]:
    text = _normalize_digits(query)
    years = [f"{year}年" for year in YEAR_PATTERN.findall(text)]
    issue_terms: list[str] = []
    for raw in ISSUE_PATTERN.findall(text):
        normalized = _normalize_digits(raw)
        issue_terms.extend([f"第{normalized}期", f"_{normalized}_npc", f"_{normalized}.pdf"])
    return unique_preserve_order(years), unique_preserve_order(issue_terms)


def _extract_explicit_exclusions(query: str) -> list[str]:
    text = _normalize_digits(query)
    exclusions: list[str] = []

    for neg in re.findall(r"不要([^，。；,.]{1,24})", text):
        exclusions.append(neg)

    if "不要第" in text and "期" in text:
        for raw in ISSUE_PATTERN.findall(text):
            normalized = _normalize_digits(raw)
            exclusions.extend([f"第{normalized}期", f"_{normalized}_npc", f"_{normalized}.pdf"])

    if any(token in text for token in ("基础版", "不要带日期后缀", "不要带日期")):
        exclusions.extend(["_20", "_19"])

    return unique_preserve_order(exclusions)


def build_retrieval_constraints(query: str, profile: QueryIntentProfile | None = None) -> RetrievalConstraints:
    original = (query or "").strip()
    profile = profile or classify_query_intent(original)

    must_include: list[str] = []
    must_not_include: list[str] = []
    prefer_doc_types: list[str] = []
    avoid_doc_types: list[str] = []

    must_include.extend(profile.article_terms)
    must_include.extend(profile.case_terms)
    must_include.extend(profile.date_terms)
    must_include.extend(profile.title_terms)
    must_include.extend(VERSION_TOKEN_PATTERN.findall(original))

    title_hard_include = False
    if profile.title_terms and profile.intent in {"article_locate", "version_effect", "doc_precise"}:
        title_hard_include = True

    if profile.wants_actual_announcement:
        prefer_doc_types.extend(["公报", "公告"])
        avoid_doc_types.extend(["案例", "合同", "申请表"])
    elif profile.prefers_summary:
        prefer_doc_types.extend(["申请表", "申报表", "办事指南"])
    elif profile.prefers_commentary:
        prefer_doc_types.extend(["指导性案例", "裁判文书", "司法解释"])

    must_not_include.extend(_extract_explicit_exclusions(original))

    gazette_years, gazette_issues = _extract_gazette_tokens(original)
    if "公报" in original:
        must_include.extend(["全国人民代表大会常务委员会公报", "公报"])
        must_include.extend(gazette_years)
        must_include.extend(gazette_issues)
        prefer_doc_types.extend(["公报"])
        avoid_doc_types.extend(["案例", "合同", "申请表", "办事指南", "裁判文书"])
        title_hard_include = True

    if profile.is_comparison or any(token in original for token in ("并列", "同时", "对比", "比较")):
        expected_doc_count = 2
    else:
        expected_doc_count = 1

    return RetrievalConstraints(
        {
            "must_include": unique_preserve_order([item for item in must_include if item]),
            "must_not_include": unique_preserve_order([item for item in must_not_include if item]),
            "prefer_doc_types": unique_preserve_order([item for item in prefer_doc_types if item]),
            "avoid_doc_types": unique_preserve_order([item for item in avoid_doc_types if item]),
            "title_hard_include": title_hard_include,
            "expected_doc_count": expected_doc_count,
            "intent": profile.intent,
        }
    )


def build_diversified_queries(
    query: str,
    profile: QueryIntentProfile | None = None,
) -> list[str]:
    original = (query or "").strip()
    if not original:
        return []

    profile = profile or classify_query_intent(original)
    planned: list[str] = []

    alias_expanded = _expand_common_legal_aliases(original)
    article_terms = ARTICLE_PATTERN.findall(original)
    case_terms = CASE_PATTERN.findall(original)
    date_terms = DATE_PATTERN.findall(original)
    doc_type_terms = _extract_doc_type_terms(original, profile)

    planned.extend(alias_expanded)

    # 对精确定位类问题优先保持“标题+锚点”检索，避免扩展词把召回带偏。
    if profile.needs_exact_match:
        for title in list(profile.title_terms)[:2]:
            planned.append(f"{title} 原文")
            for article in article_terms[:2]:
                planned.append(f"{title} {article}")
                planned.append(f"{title} {article} 条文原文")
        for case in case_terms[:2]:
            planned.append(f"{case} 原文")
        for d in date_terms[:2]:
            planned.append(f"{original} {d}")

    if profile.intent == "article_locate":
        for article in article_terms[:2]:
            for q in alias_expanded[:2]:
                planned.append(f"{q} {article} 条文原文")
                planned.append(f"{q} {article} 原文位置")
        for title in list(profile.title_terms)[:2]:
            for article in article_terms[:2]:
                planned.append(f"《{title}》 {article} 原文")
                planned.append(f"{title} 全文 {article} 条文")
                planned.append(f"中华人民共和国{title} {article}")

    if profile.intent == "procedure_extract":
        planned.append(f"{original} 申请条件 申请材料 办理流程 办理时限 受理机关")
        planned.append(f"{original} 办事指南 许可目录 申报表")

    if profile.intent in {"penalty_elements", "version_effect"} and not profile.needs_exact_match:
        planned.append(f"{original} 适用条件 构成要件 处罚")
        planned.append(f"{original} 生效 修订 现行")

    if profile.intent in {"case_holding", "case_reasoning"}:
        for case in case_terms[:2]:
            planned.append(f"{case} 裁判要旨 争议焦点 裁判理由")
        planned.append(f"{original} 基本案情 裁判理由 结论")

    if profile.intent == "contract_clause":
        planned.append(f"{original} 合同目的 权利义务 违约责任 解除条件 争议解决")

    if profile.intent == "comparison":
        planned.append(f"{original} 对比 差异 适用范围")
        planned.append(f"{original} 法条 比较")

    if "公报" in original:
        years, issues = _extract_gazette_tokens(original)
        planned.append(f"{original} 全国人民代表大会常务委员会公报")
        if years and issues:
            for year in years[:2]:
                for issue in issues[:4]:
                    planned.append(f"全国人民代表大会常务委员会公报 {year} {issue}")
                    planned.append(f"全国人民代表大会常务委员会公报 {year} {issue} npc")
        elif years:
            for year in years[:2]:
                planned.append(f"全国人民代表大会常务委员会公报 {year}")
        elif issues:
            for issue in issues[:4]:
                planned.append(f"全国人民代表大会常务委员会公报 {issue}")

    for dt in doc_type_terms[:4]:
        planned.append(f"{original} {dt}")

    for d in date_terms[:2]:
        planned.append(f"{original} {d} 生效 修订")

    return unique_preserve_order(planned)[:8]


def chunk_identity(chunk: dict[str, Any]) -> str:
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    content = re.sub(r"\s+", " ", str(chunk.get("content_with_weight", ""))).strip()
    return f"{chunk.get('document_id', '')}::{content[:160]}"


def chunk_document_key(chunk: dict[str, Any]) -> str:
    document_name = str(chunk.get("document_name", "")).strip().lower()
    if document_name:
        return document_name
    document_id = str(chunk.get("document_id", "")).strip().lower()
    if document_id:
        return document_id
    return chunk_identity(chunk)


def normalize_match_text(value: str) -> str:
    return re.sub(r"\s+", "", (value or "").lower())


def _metadata_haystack(chunk: dict[str, Any]) -> str:
    return normalize_match_text(
        " ".join(
            str(chunk.get(key, ""))
            for key in (
                "document_name",
                "doc_type",
                "authority",
                "legal_domain",
                "article_anchors",
                "case_anchors",
                "procedure_anchors",
                "contract_anchors",
                "content_with_weight",
            )
        )
    )


def _chunk_match_boost(chunk: dict[str, Any], profile: QueryIntentProfile) -> int:
    haystack = _metadata_haystack(chunk)
    score = 0
    for term in profile.doc_type_terms:
        if normalize_match_text(term) in haystack:
            score += 4
    for term in profile.authority_terms:
        if normalize_match_text(term) in haystack:
            score += 3
    for term in profile.article_terms:
        if normalize_match_text(term) in haystack:
            score += 6
    for term in profile.case_terms:
        if normalize_match_text(term) in haystack:
            score += 6
    for term in profile.date_terms:
        if normalize_match_text(term) in haystack:
            score += 2
    for term in profile.title_terms:
        if normalize_match_text(term) in haystack:
            score += 6
    return score


def diversify_chunks_by_document(
    chunks: list[dict[str, Any]],
    planned_query: str,
    max_per_document: int = 2,
) -> list[dict[str, Any]]:
    if len(chunks) <= 2:
        return list(chunks)

    grouped: dict[str, list[dict[str, Any]]] = {}
    ordered_keys: list[str] = []
    for chunk in chunks:
        key = chunk_document_key(chunk)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(chunk)

    diversified: list[dict[str, Any]] = []
    depth = 0
    while True:
        appended = False
        for key in ordered_keys:
            bucket = grouped[key]
            if depth >= len(bucket) or depth >= max_per_document:
                continue
            diversified.append(bucket[depth])
            appended = True
        if not appended:
            break
        depth += 1

    if len(diversified) >= len(chunks):
        return diversified[: len(chunks)]

    seen = {chunk_identity(chunk) for chunk in diversified}
    for chunk in chunks:
        identity = chunk_identity(chunk)
        if identity in seen:
            continue
        diversified.append(chunk)
        seen.add(identity)

    return diversified[: len(chunks)]


def filter_chunks_by_query_context(
    chunks: list[dict[str, Any]],
    company: str,
    planned_query: str,
    profile: QueryIntentProfile | None = None,
) -> list[dict[str, Any]]:
    profile = profile or classify_query_intent(planned_query)
    ranked = sorted(
        chunks,
        key=lambda chunk: (
            -_chunk_match_boost(chunk, profile),
            -float(chunk.get("_fusion_score", chunk.get("fusion_score", 0.0)) or 0.0),
            -float(chunk.get("_best_similarity", chunk.get("similarity", 0.0)) or 0.0),
            int(chunk.get("_best_rank", chunk.get("best_rank", chunk.get("source_rank", chunk.get("rank", 999)))) or 999),
        ),
    )
    return diversify_chunks_by_document(ranked, planned_query)


def merge_ranked_chunks(
    chunk_lists: list[list[dict[str, Any]]],
    limit: int,
    company: str = "",
    planned_query: str = "",
    profile: QueryIntentProfile | None = None,
) -> list[dict[str, Any]]:
    merged_by_identity: dict[str, dict[str, Any]] = {}
    ordered_identities: list[str] = []

    for list_index, chunk_list in enumerate(chunk_lists):
        for chunk in chunk_list:
            identity = chunk_identity(chunk)
            source_rank = int(chunk.get("source_rank", chunk.get("rank", 999)) or 999)
            similarity = float(chunk.get("similarity", 0.0) or 0.0)
            lexical_score = float(chunk.get("term_similarity", 0.0) or 0.0)
            vector_score = float(chunk.get("vector_similarity", 0.0) or 0.0)
            # keep lexical dominance for precise legal entity lookup
            fusion_score = (
                (1.2 / (60 + max(1, source_rank)))
                + 0.75 * max(0.0, lexical_score)
                + 0.20 * max(0.0, vector_score)
                + 0.05 * max(0.0, similarity)
            )
            if identity not in merged_by_identity:
                merged = dict(chunk)
                merged["source_rank"] = source_rank
                merged["_appearance_count"] = 1
                merged["_best_rank"] = source_rank
                merged["_best_similarity"] = similarity
                merged["_fusion_score"] = fusion_score
                merged["_first_list_index"] = list_index
                merged_by_identity[identity] = merged
                ordered_identities.append(identity)
                continue

            existing = merged_by_identity[identity]
            existing["_appearance_count"] = int(existing.get("_appearance_count", 1)) + 1
            existing["_best_rank"] = min(int(existing.get("_best_rank", source_rank) or source_rank), source_rank)
            existing["_best_similarity"] = max(float(existing.get("_best_similarity", 0.0) or 0.0), similarity)
            existing["_fusion_score"] = float(existing.get("_fusion_score", 0.0) or 0.0) + fusion_score

    merged = [merged_by_identity[identity] for identity in ordered_identities]
    if planned_query or company:
        merged = filter_chunks_by_query_context(merged, company, planned_query, profile=profile)

    ranked: list[dict[str, Any]] = []
    for display_rank, chunk in enumerate(merged[:limit], start=1):
        item = dict(chunk)
        item.pop("_first_list_index", None)
        item["id"] = str(item.get("chunk_id") or item.get("id") or chunk_identity(item))
        item["display_rank"] = display_rank
        item["merge_rank"] = display_rank
        ranked.append(item)
    return ranked
