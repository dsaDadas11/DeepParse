from __future__ import annotations

import re
import time
from pathlib import Path
from threading import Lock
from typing import Any

from pypdf import PdfReader

from runtime_config import (
    ENABLE_FALLBACK_ROUTE,
    ENABLE_LEGAL_METADATA_HARD_FILTER,
    ENABLE_LEGAL_METADATA_ROUTE,
    ENABLE_LEGAL_METADATA_SCORING,
    ENABLE_RETRIEVAL_TRACE,
    RETRIEVAL_ROUTE_MODE,
)
from service.core.conversation import extract_company_candidate, rewrite_question_with_history

from service.core.retrieval_intent import classify_query_intent
from service.core.rag.nlp.search_v2 import index_name
from service.core.rag.utils.doc_store_conn import OrderByExpr
from service.core.rag_config import (
    DEFAULT_TOP_K,
    RUNTIME_SCORE_DOC_TYPE_AVOID_PENALTY,
    RUNTIME_SCORE_DOC_TYPE_PREFER_BONUS,
    RUNTIME_SCORE_MUST_INCLUDE_FULL_COVERAGE_BONUS,
    RUNTIME_SCORE_MUST_INCLUDE_PER_MATCH,
    RUNTIME_SCORE_MUST_NOT_INCLUDE_HARD_PENALTY,
    RUNTIME_SCORE_TITLE_HARD_BONUS,
    RUNTIME_SCORE_TITLE_MISS_PENALTY,
    RUNTIME_SCORE_LEGAL_METADATA_STRONG_BONUS,
    RUNTIME_SCORE_LEGAL_METADATA_UNIT,
)
from utils import logger

from .legal_hybrid_retrieval import build_index as build_legal_index
from .legal_hybrid_retrieval import retrieve as legal_retrieve
from .legal_query_parser import (
    LegalQuerySlots,
    coverage_aware_rerank,
    legal_hard_filter,
    legal_metadata_score,
    parse_legal_query,
    rank_corpus_documents,
)
from .retrieval import dealer, retrieve_content as legacy_retrieve_content
from .retrieval_query_planner import (
    ARTICLE_PATTERN,
    DATE_PATTERN,
    ISSUE_PATTERN,
    build_diversified_queries,
    build_retrieval_constraints,
    filter_chunks_by_query_context,
    merge_ranked_chunks,
)

DEFAULT_FIELDS = [
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
    "doc_type_kwd",
    "authority_kwd",
    "legal_domain_kwd",
    "effective_or_revision_date_kwd",
    "version_scope_kwd",
    "article_anchors_kwd",
    "case_anchors_kwd",
    "procedure_anchors_kwd",
    "contract_anchors_kwd",
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
]

ANNOUNCEMENT_REQUEST_TOKENS = (
    "法条原文",
    "条文原文",
    "法律原文",
    "司法解释原文",
    "指导性案例原文",
    "完整条文",
    "全文",
    "原文",
    "不要解读",
    "不看解读",
    "仅要法条",
    "仅要条文",
)
SUMMARY_REQUEST_TOKENS = (
    "摘要",
    "表格",
    "办事流程",
    "申请材料",
    "受理条件",
    "办理时限",
    "不用全文",
    "不看全文",
    "不用整本",
    "不看整本",
)
LEADING_REQUEST_PATTERN = re.compile(
    r"^(?:我想先看|我想看|我需要|我只想|先把|先给我|先看|帮我找|帮我看|给我|把|发我|给我发|告诉我|我先看)"
)
NEGATIVE_COMMENTARY_PATTERN = re.compile(r"(?:不要|不看|不用).{0,8}(?:解读|评述|点评|说明)")
PERIOD_PATTERN = re.compile(r"(?P<year>20\d{2})年?(?:(?P<month>\d{1,2})月)?(?:(?P<day>\d{1,2})日)?|(?P<compact>20\d{6})")


def _normalize_digits(text: str) -> str:
    return (text or "").translate(str.maketrans("0123456789", "0123456789"))


LEGAL_CORPUS_LIST_PATH = Path(__file__).resolve().parents[2] / "sample_data" / "pdf_list.txt"
_LEGAL_PIPELINE_READY = False
_LEGAL_PIPELINE_LOCK = Lock()


def _safe_read_pdf_text(file_path: Path) -> str:
    try:
        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(page_text)
        return "\n".join(pages)
    except Exception as exc:
        logger.warning("Failed to read legal corpus PDF %s: %s", file_path, exc)
        return ""


def _resolve_legal_pdf_path(raw_path: str, list_file: Path) -> Path:
    raw = (raw_path or "").strip().strip('"').strip("'").lstrip("\ufeff")
    if not raw:
        return Path()

    candidate = Path(raw)
    # Windows absolute path in list file (e.g. C:\\...): map to container sample_data/pdfs
    if re.match(r"^[A-Za-z]:[\\/]", raw):
        filename = Path(raw.replace("\\", "/")).name
        return (list_file.parent / "pdfs" / filename).resolve()

    if candidate.is_absolute():
        return candidate
    return (list_file.parent / candidate).resolve()


def _load_legal_documents(list_file: Path) -> list[dict[str, Any]]:
    if not list_file.exists() or not list_file.is_file():
        logger.warning("Legal corpus list file not found: %s", list_file)
        return []

    try:
        raw_lines = list_file.read_text(encoding="utf-8-sig").splitlines()
    except Exception as exc:
        logger.warning("Failed to read legal corpus list file %s: %s", list_file, exc)
        return []

    pdf_paths: list[Path] = []
    for line in raw_lines:
        pdf_path = _resolve_legal_pdf_path(line, list_file)
        if not str(pdf_path):
            continue
        pdf_paths.append(pdf_path)

    documents: list[dict[str, Any]] = []
    for pdf_file in pdf_paths:
        try:
            is_valid = pdf_file.exists() and pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf"
        except OSError:
            is_valid = False

        if not is_valid:
            logger.warning("Skipping invalid legal corpus PDF path: %s", pdf_file)
            continue

        text = _safe_read_pdf_text(pdf_file)
        if not text:
            continue
        documents.append(
            {
                "doc_id": pdf_file.stem,
                "file_name": pdf_file.name,
                "full_title": pdf_file.stem,
                "source_path": str(pdf_file),
                "text": text,
                "tables": [],
            }
        )

    logger.info("Loaded %s legal documents from list: %s", len(documents), list_file)
    return documents


def _ensure_legal_pipeline_ready() -> bool:
    global _LEGAL_PIPELINE_READY
    if _LEGAL_PIPELINE_READY:
        return True

    with _LEGAL_PIPELINE_LOCK:
        if _LEGAL_PIPELINE_READY:
            return True
        documents = _load_legal_documents(LEGAL_CORPUS_LIST_PATH)
        if not documents:
            logger.warning("No legal documents loaded; legal retrieval pipeline disabled.")
            return False
        try:
            build_legal_index(documents, use_sentence_transformer=False)
            _LEGAL_PIPELINE_READY = True
            logger.info("Legal retrieval pipeline initialized.")
            return True
        except Exception as exc:
            logger.warning("Failed to initialize legal retrieval pipeline: %s", exc)
            return False


def _first_page(page_num_int: Any) -> int | None:
    if isinstance(page_num_int, list) and page_num_int:
        try:
            return int(page_num_int[0])
        except (TypeError, ValueError):
            return None
    try:
        return int(page_num_int)
    except (TypeError, ValueError):
        return None


def _build_chunk(field: dict[str, Any], index_names: str, rank: int, **scores: Any) -> dict[str, Any]:
    positions = field.get("position_int", []) or []
    if not positions:
        positions = [[page] for page in field.get("page_num_int", [])]
    page_num_int = field.get("page_num_int", []) or []
    chunk_id = field.get("chunk_id") or field.get("id") or f"{field.get('doc_id', 'N/A')}::{rank}"
    source_rank = int(scores.get("source_rank", rank) or rank)
    display_rank = int(scores.get("display_rank", rank) or rank)
    return {
        "id": str(chunk_id),
        "rank": source_rank,
        "source_rank": source_rank,
        "display_rank": display_rank,
        "index_name": index_names,
        "document_id": field.get("doc_id", "N/A"),
        "chunk_id": chunk_id,
        "document_name": field.get("docnm_kwd", "").split("/")[-1],
        "content_with_weight": field.get("content_with_weight", "N/A"),
        "company": field.get("company_kwd", ""),
        "report_period": field.get("report_period_kwd", ""),
        "report_type": field.get("report_type_kwd", ""),
        "source": field.get("source_kwd", ""),
        "doc_type": field.get("doc_type_kwd", ""),
        "authority": field.get("authority_kwd", ""),
        "legal_domain": field.get("legal_domain_kwd", ""),
        "effective_or_revision_date": field.get("effective_or_revision_date_kwd", ""),
        "version_scope": field.get("version_scope_kwd", ""),
        "article_anchors": list(field.get("article_anchors_kwd", []) or []),
        "case_anchors": list(field.get("case_anchors_kwd", []) or []),
        "procedure_anchors": list(field.get("procedure_anchors_kwd", []) or []),
        "contract_anchors": list(field.get("contract_anchors_kwd", []) or []),
        "table_dense_int": int(field.get("table_dense_int", 0) or 0),
        "table_headers_kwd": list(field.get("table_headers_kwd", []) or []),
        "table_rows_kwd": list(field.get("table_rows_kwd", []) or []),
        "positions": positions,
        "page_num_int": page_num_int,
        "page_num": _first_page(page_num_int),
        "similarity": float(scores.get("similarity", 0.0) or 0.0),
        "term_similarity": float(scores.get("term_similarity", 0.0) or 0.0),
        "vector_similarity": float(scores.get("vector_similarity", 0.0) or 0.0),
    }


def _raw_hits_to_chunks(res: dict, src_fields: list[str], index_names: str) -> list[dict[str, Any]]:
    ids = dealer.dataStore.getChunkIds(res)
    fields = dealer.dataStore.getFields(res, src_fields)
    chunks: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ids, start=1):
        field = dict(fields.get(chunk_id, {}))
        if not field:
            continue
        field["chunk_id"] = chunk_id
        chunks.append(_build_chunk(field, index_names, rank))
    return chunks


def _dense_fallback(index_names: str, question: str, page_size: int) -> list[dict[str, Any]]:
    src = list(DEFAULT_FIELDS)
    order_by = OrderByExpr()
    filters = {"available_int": 1}
    match_dense = dealer.get_vector(question, None, 1024, 0.1)
    vector = match_dense.embedding_data
    src.append(f"q_{len(vector)}_vec")
    res = dealer.dataStore.search(
        src,
        [],
        filters,
        [match_dense],
        order_by,
        0,
        page_size,
        [index_name(index_names)],
        None,
    )
    return _raw_hits_to_chunks(res, src, index_names)


def _strip_leading_request_phrases(query: str) -> str:
    text = (query or "").strip()
    while True:
        stripped = LEADING_REQUEST_PATTERN.sub("", text).strip()
        if stripped == text:
            return text
        text = stripped


def _announcement_period_aliases(query: str) -> list[str]:
    aliases: list[str] = []
    for match in PERIOD_PATTERN.finditer(query or ""):
        raw = (match.group(0) or "").strip()
        year = (match.group("year") or "").strip()
        month = (match.group("month") or "").strip()
        day = (match.group("day") or "").strip()
        compact = (match.group("compact") or "").strip()

        if raw:
            aliases.append(raw)
        if compact:
            aliases.append(compact)
            if len(compact) == 8:
                aliases.append(f"{compact[:4]}年{compact[4:6]}月{compact[6:]}日")
        if year:
            aliases.append(year)
            aliases.append(f"{year}年")
            if month:
                aliases.append(f"{year}年{month}月")
                aliases.append(f"{year}{month.zfill(2)}")
            if month and day:
                aliases.append(f"{year}年{month}月{day}日")
                aliases.append(f"{year}{month.zfill(2)}{day.zfill(2)}")

    return list(dict.fromkeys(alias for alias in aliases if alias))


def _sanitize_announcement_company(query: str) -> str:
    text = _strip_leading_request_phrases(query)
    text = PERIOD_PATTERN.sub("", text, count=1)
    text = re.split(
        r"(?:的?(?:法条原文|条文原文|法律原文|司法解释原文|案例原文|原文|全文|摘要|办事流程|申请材料|受理条件|办理时限|解读|评述|点评|核心内容|适用条件)|[,，。；])",
        text,
        maxsplit=1,
    )[0]
    return text.strip()


def _looks_like_actual_announcement_query(query: str, profile: Any) -> bool:
    text = query or ""
    if getattr(profile, "prefers_summary", False):
        return False
    if getattr(profile, "wants_actual_announcement", False):
        return True
    if not any(token in text for token in ANNOUNCEMENT_REQUEST_TOKENS):
        return False
    if any(token in text for token in SUMMARY_REQUEST_TOKENS):
        return False
    if NEGATIVE_COMMENTARY_PATTERN.search(text):
        return True
    if any(token in text for token in ("点评", "研报")):
        return False
    return True


def _is_announcement_like_reference(chunk: dict[str, Any]) -> bool:
    document_name = str(chunk.get("document_name", "") or chunk.get("docnm_kwd", ""))
    legal_raw_markers = ("法", "条例", "规定", "司法解释", "指导性案例", "办事指南", "合同", "公报")
    exclude_markers = ("摘要", "解读", "评述", "点评")
    return any(marker in document_name for marker in legal_raw_markers) and not any(
        marker in document_name for marker in exclude_markers
    )


def _extract_gazette_issue_keys(query: str) -> list[str]:
    text = _normalize_digits(query or "")
    keys: list[str] = []
    for year, issue in re.findall(r"(20\d{2})年?第?([0-9一二三四五六七八九十]{1,2})期", text):
        zh_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
        issue_n = zh_map.get(issue, issue)
        keys.append(f"{year}_{issue_n}")
    return list(dict.fromkeys(keys))


def _is_gazette_query(query: str) -> bool:
    text = _normalize_digits(query or "")
    if "公报" not in text:
        return False
    return bool(re.search(r"20\d{2}", text) or ISSUE_PATTERN.search(text))


def _looks_like_gazette_reference(chunk: dict[str, Any]) -> bool:
    document_name = str(chunk.get("document_name", "") or chunk.get("docnm_kwd", ""))
    return "公报" in document_name and "npc" in document_name.lower()


def _prepend_unique_references(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in list(primary) + list(secondary):
        identity = str(chunk.get("chunk_id") or chunk.get("id") or f"{chunk.get('document_name', '')}:{chunk.get('page_num')}")
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged


def _ensure_distinct_docs(chunks: list[dict[str, Any]], required_docs: int, limit: int) -> list[dict[str, Any]]:
    if required_docs <= 1:
        return chunks[:limit]
    selected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for chunk in chunks:
        doc = re.sub(r"\s+", "", str(chunk.get("document_name", "") or chunk.get("document_id", "")).lower())
        if doc and doc not in seen_docs:
            selected.append(chunk)
            seen_docs.add(doc)
        if len(seen_docs) >= required_docs:
            break
    for chunk in chunks:
        if len(selected) >= limit:
            break
        if chunk in selected:
            continue
        selected.append(chunk)
    return selected[:limit]


def _prepend_unique_documents(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    seen_chunks: set[str] = set()
    for chunk in list(primary) + list(secondary):
        doc_key = re.sub(r"\s+", "", str(chunk.get("document_name", "") or chunk.get("document_id", "")).lower())
        chunk_key = str(chunk.get("chunk_id") or chunk.get("id") or f"{doc_key}:{chunk.get('page_num')}")
        if doc_key:
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)
        elif chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged


def _metadata_lookup_references(index_names: str, slots: LegalQuerySlots, limit: int) -> list[dict[str, Any]]:
    if not ENABLE_LEGAL_METADATA_ROUTE or not slots.strong_constraint or slots.route == "law_article_lookup":
        return []

    hits = rank_corpus_documents(LEGAL_CORPUS_LIST_PATH, slots, max(limit, slots.expected_doc_count))
    chunks: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits, start=1):
        file_name = str(hit.get("file_name", ""))
        if not file_name:
            continue
        field = {
            "doc_id": hit.get("doc_id") or Path(file_name).stem,
            "docnm_kwd": file_name,
            "content_with_weight": hit.get("content") or Path(file_name).stem,
            "page_num_int": [],
            "position_int": [],
            "chunk_id": f"legal_metadata::{Path(file_name).stem}",
            "company_kwd": "",
            "report_period_kwd": "",
            "report_type_kwd": "",
            "source_kwd": "legal_metadata",
            "doc_type_kwd": "法律元数据",
            "authority_kwd": "",
            "legal_domain_kwd": "",
            "effective_or_revision_date_kwd": "",
            "version_scope_kwd": "",
            "article_anchors_kwd": [],
            "case_anchors_kwd": [],
            "procedure_anchors_kwd": [],
            "contract_anchors_kwd": [],
            "table_dense_int": 0,
            "table_headers_kwd": [],
            "table_rows_kwd": [],
        }
        chunks.append(
            _build_chunk(
                field,
                index_names,
                rank,
                source_rank=rank,
                display_rank=rank,
                similarity=float(hit.get("score", 0.0) or 0.0),
                term_similarity=float(hit.get("score", 0.0) or 0.0),
                vector_similarity=0.0,
            )
        )
    return chunks[:limit]


def _can_short_circuit_metadata(slots: LegalQuerySlots, references: list[dict[str, Any]]) -> bool:
    if not references or slots.route == "law_article_lookup":
        return False
    if slots.route in {"gazette_issue_lookup", "same_name_disambiguation", "version_conflict_lookup", "version_diff_lookup"}:
        return len(references) >= min(slots.expected_doc_count, DEFAULT_TOP_K)
    if slots.route == "version_lookup":
        return bool(slots.title_terms and slots.version_dates)
    return False


def _announcement_docname_fallback(index_names: str, question: str, page_size: int) -> list[dict[str, Any]]:
    title_hint = _sanitize_announcement_company(question)
    if not title_hint:
        return []

    date_aliases = _announcement_period_aliases(question)
    should_filters: list[dict[str, Any]] = []

    should_filters.append({"match_phrase": {"docnm_kwd": title_hint}})
    should_filters.append({"match_phrase": {"content_with_weight": title_hint}})

    for alias in date_aliases[:6]:
        should_filters.append({"match_phrase": {"docnm_kwd": alias}})
        should_filters.append({"match_phrase": {"content_with_weight": alias}})

    body = {
        "query": {
            "bool": {
                "filter": [{"term": {"available_int": 1}}],
                "should": should_filters,
                "minimum_should_match": 1,
            }
        },
        "_source": DEFAULT_FIELDS,
        "sort": [
            {"_score": {"order": "desc"}},
            {"page_num_int": {"order": "asc", "unmapped_type": "integer"}},
            {"chunk_id": {"order": "asc", "unmapped_type": "keyword"}},
        ],
        "size": page_size,
    }

    response = dealer.dataStore.es.search(
        index=index_name(index_names),
        body=body,
        timeout="120s",
        track_total_hits=False,
    )
    chunks = _raw_hits_to_chunks(response, DEFAULT_FIELDS, index_names)
    return chunks


def _prefer_legacy_route(question: str) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    normalized = (text or "").translate(str.maketrans("0123456789", "0123456789"))
    if ISSUE_PATTERN.search(normalized):
        return True
    if "公报" in text:
        return True
    return False


def _retrieve_single(index_names: str, question: str, page_size: int) -> tuple[list[dict[str, Any]], str]:
    route_mode = (RETRIEVAL_ROUTE_MODE or "auto").lower()

    def _legacy_retrieve() -> list[dict[str, Any]]:
        results = dealer.retrieval(
            question=question,
            embd_mdl=None,
            tenant_ids=index_names,
            kb_ids=None,
            vector_similarity_weight=0.6,
            page=1,
            page_size=page_size,
        )
        extracted_data: list[dict[str, Any]] = []
        for rank, chunk in enumerate(results["chunks"], start=1):
            field = dict(chunk)
            extracted_data.append(
                _build_chunk(
                    field,
                    index_names,
                    rank,
                    source_rank=chunk.get("source_rank", chunk.get("rank", rank)),
                    display_rank=chunk.get("display_rank", rank),
                    similarity=chunk.get("similarity", 0.0),
                    term_similarity=chunk.get("term_similarity", 0.0),
                    vector_similarity=chunk.get("vector_similarity", 0.0),
                )
            )
        return extracted_data

    if route_mode == "legacy" or (route_mode == "auto" and _prefer_legacy_route(question)):
        try:
            legacy_chunks = _legacy_retrieve()
            if legacy_chunks:
                return legacy_chunks, "legacy"
        except AssertionError:
            logger.warning("Hybrid retrieval assertion failed; falling back to dense retrieval.")
            return _dense_fallback(index_names, question, page_size), "dense_fallback"

    if route_mode in {"hybrid", "auto"} and _ensure_legal_pipeline_ready():
        try:
            legal_results = legal_retrieve(question, top_k=page_size)
            extracted_data: list[dict[str, Any]] = []
            for rank, item in enumerate(legal_results, start=1):
                metadata = dict(item.get("metadata") or {})
                score_breakdown = dict(item.get("score_breakdown") or {})
                field = {
                    "doc_id": metadata.get("doc_id", "N/A"),
                    "docnm_kwd": metadata.get("file_name", ""),
                    "content_with_weight": item.get("text", "N/A"),
                    "page_num_int": [],
                    "position_int": [],
                    "chunk_id": item.get("chunk_id", f"legal::{rank}"),
                    "company_kwd": "",
                    "report_period_kwd": "",
                    "report_type_kwd": "",
                    "source_kwd": "",
                    "doc_type_kwd": "法律",
                    "authority_kwd": "",
                    "legal_domain_kwd": "",
                    "effective_or_revision_date_kwd": metadata.get("version_date", ""),
                    "version_scope_kwd": metadata.get("edition_tag", ""),
                    "article_anchors_kwd": [],
                    "case_anchors_kwd": [],
                    "procedure_anchors_kwd": [],
                    "contract_anchors_kwd": [],
                    "table_dense_int": 0,
                    "table_headers_kwd": [],
                    "table_rows_kwd": [],
                }
                extracted_data.append(
                    _build_chunk(
                        field,
                        index_names,
                        rank,
                        source_rank=rank,
                        display_rank=rank,
                        similarity=score_breakdown.get("final_score", 0.0),
                        term_similarity=score_breakdown.get("bm25_score", 0.0),
                        vector_similarity=score_breakdown.get("vector_score", 0.0),
                    )
                )
            if extracted_data:
                return extracted_data, "hybrid"
        except Exception as exc:
            logger.warning("Legal retrieval failed; fallback to existing retrieval. error=%s", exc)

    if route_mode in {"legacy", "auto", "hybrid"}:
        try:
            legacy_chunks = _legacy_retrieve()
            return legacy_chunks, "legacy"
        except AssertionError:
            logger.warning("Hybrid retrieval assertion failed; falling back to dense retrieval.")
            return _dense_fallback(index_names, question, page_size), "dense_fallback"

    return _dense_fallback(index_names, question, page_size), "dense_fallback"


def _contains_anchor(chunk: dict[str, Any], anchors: tuple[str, ...]) -> bool:
    if not anchors:
        return True
    text = str(chunk.get("content_with_weight", "") or "")
    doc = str(chunk.get("document_name", "") or "")
    return any(anchor and (anchor in text or anchor in doc) for anchor in anchors)


def _normalize_matching_text(text: str) -> str:
    if not text:
        return ""
    normalized = str(text)
    full_to_half = str.maketrans(
        "0123456789（）《》【】：，。；",
        "0123456789()<>[]:,.;",
    )
    normalized = normalized.translate(full_to_half)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _matches_constraint(chunk: dict[str, Any], term: str) -> bool:
    if not term:
        return False
    text = str(chunk.get("content_with_weight", "") or "")
    doc = str(chunk.get("document_name", "") or "")
    haystack = _normalize_matching_text(f"{doc} {text}")

    needle = _normalize_matching_text(term)
    if not needle:
        return False

    if needle in haystack:
        return True

    simplified = needle.replace("《", "").replace(">", "").replace("<", "").replace("》", "")
    if simplified and simplified in haystack:
        return True
    return False


def _apply_constraints(chunks: list[dict[str, Any]], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    must_include = list(constraints.get("must_include") or [])
    must_not_include = list(constraints.get("must_not_include") or [])

    filtered: list[dict[str, Any]] = []
    for chunk in chunks:
        if must_not_include and any(_matches_constraint(chunk, term) for term in must_not_include):
            continue
        filtered.append(chunk)

    if not must_include:
        return filtered

    matched_counts: list[tuple[int, dict[str, Any]]] = []
    for chunk in filtered:
        matched_count = sum(1 for term in must_include if _matches_constraint(chunk, term))
        matched_counts.append((matched_count, chunk))

    hard = [chunk for matched_count, chunk in matched_counts if matched_count > 0]
    if not hard:
        return filtered

    ranked_hard = sorted(
        (item for item in matched_counts if item[0] > 0),
        key=lambda item: item[0],
        reverse=True,
    )
    hard_sorted = [chunk for _, chunk in ranked_hard]
    soft = [chunk for chunk in filtered if chunk not in hard_sorted]
    return hard_sorted + soft


def _enforce_doc_diversity(chunks: list[dict[str, Any]], expected_doc_count: int, limit: int) -> list[dict[str, Any]]:
    if expected_doc_count <= 1:
        return chunks[:limit]

    selected: list[dict[str, Any]] = []
    seen_docs: set[str] = set()
    for chunk in chunks:
        doc = str(chunk.get("document_name", "") or chunk.get("document_id", ""))
        if doc and doc not in seen_docs:
            selected.append(chunk)
            seen_docs.add(doc)
        if len(seen_docs) >= expected_doc_count:
            break

    for chunk in chunks:
        if len(selected) >= limit:
            break
        if chunk in selected:
            continue
        selected.append(chunk)

    return selected[:limit]


def _evidence_rank(
    chunks: list[dict[str, Any]],
    profile: Any,
    constraints: dict[str, Any],
    legal_slots: LegalQuerySlots,
) -> list[dict[str, Any]]:
    constrained_chunks = _apply_constraints(chunks, constraints)
    if ENABLE_LEGAL_METADATA_SCORING and ENABLE_LEGAL_METADATA_HARD_FILTER:
        constrained_chunks = legal_hard_filter(constrained_chunks, legal_slots)

    prefer_doc_types = tuple(constraints.get("prefer_doc_types") or [])
    avoid_doc_types = tuple(constraints.get("avoid_doc_types") or [])
    must_include_terms = tuple(constraints.get("must_include") or [])
    must_not_include_terms = tuple(constraints.get("must_not_include") or [])
    title_terms = tuple(getattr(profile, "title_terms", ()) or ())
    title_hard_include = bool(constraints.get("title_hard_include", False))

    def _doc_type_score(chunk: dict[str, Any]) -> int:
        doc_name = str(chunk.get("document_name", "") or "")
        score = 0
        for token in prefer_doc_types:
            if token and token in doc_name:
                score += RUNTIME_SCORE_DOC_TYPE_PREFER_BONUS
        for token in avoid_doc_types:
            if token and token in doc_name:
                score -= RUNTIME_SCORE_DOC_TYPE_AVOID_PENALTY
        return score

    def _title_score(chunk: dict[str, Any]) -> int:
        if not title_terms:
            return 0
        title_hits = sum(1 for term in title_terms if _matches_constraint(chunk, term))
        if title_hard_include:
            return RUNTIME_SCORE_TITLE_HARD_BONUS * title_hits if title_hits > 0 else -RUNTIME_SCORE_TITLE_MISS_PENALTY
        return title_hits * 2

    def score(chunk: dict[str, Any]) -> tuple[int, int, int, int, float, int]:
        hard = 0
        if _contains_anchor(chunk, tuple(profile.article_terms)):
            hard += 2 if profile.article_terms else 0
        if _contains_anchor(chunk, tuple(profile.case_terms)):
            hard += 2 if profile.case_terms else 0
        if _contains_anchor(chunk, tuple(profile.date_terms)):
            hard += 1 if profile.date_terms else 0

        matched_include = sum(1 for t in must_include_terms if _matches_constraint(chunk, t))
        if matched_include:
            hard += matched_include * RUNTIME_SCORE_MUST_INCLUDE_PER_MATCH
            if matched_include == len(must_include_terms):
                hard += RUNTIME_SCORE_MUST_INCLUDE_FULL_COVERAGE_BONUS

        if must_not_include_terms and any(_matches_constraint(chunk, t) for t in must_not_include_terms):
            hard -= RUNTIME_SCORE_MUST_NOT_INCLUDE_HARD_PENALTY

        title_boost = _title_score(chunk)
        doc_type_boost = _doc_type_score(chunk)
        metadata_boost = 0
        if ENABLE_LEGAL_METADATA_SCORING:
            metadata_boost = legal_metadata_score(chunk, legal_slots) * RUNTIME_SCORE_LEGAL_METADATA_UNIT
            if metadata_boost >= 100:
                hard += RUNTIME_SCORE_LEGAL_METADATA_STRONG_BONUS
        sim = float(chunk.get("_fusion_score", chunk.get("fusion_score", 0.0)) or 0.0)
        src_rank = int(chunk.get("source_rank", chunk.get("rank", 999)) or 999)
        return hard, metadata_boost, title_boost, doc_type_boost, sim, -src_rank

    ranked = sorted(constrained_chunks, key=score, reverse=True)
    if ENABLE_LEGAL_METADATA_SCORING:
        ranked = coverage_aware_rerank(ranked, legal_slots, len(ranked))
    expected_doc_count = max(
        int(constraints.get("expected_doc_count", 1) or 1),
        int(getattr(legal_slots, "expected_doc_count", 1) or 1),
    )
    return _enforce_doc_diversity(ranked, expected_doc_count, len(ranked))


def build_retrieval_context(
    index_names: str,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    history_turns: list[dict[str, Any]] | None = None,
    standalone_query: str | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    if history_turns and not standalone_query:
        standalone_query, _ = rewrite_question_with_history(question, history_turns)

    effective_query = (standalone_query or question or "").strip()
    if not effective_query:
        return {
            "question": question,
            "standalone_query": "",
            "planned_queries": [],
            "references": [],
            "trace": {
                "question": question,
                "standalone_query": "",
                "planned_queries": [],
                "per_query": [],
                "fallback_used": False,
                "returned_count": 0,
                "total_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        }

    legal_slots = parse_legal_query(effective_query)
    effective_profile = classify_query_intent(effective_query)
    constraints = build_retrieval_constraints(effective_query, profile=effective_profile)
    metadata_references: list[dict[str, Any]] = []
    metadata_route_used = False
    metadata_route_count = 0
    if legal_slots.strong_constraint and ENABLE_LEGAL_METADATA_ROUTE:
        metadata_references = _metadata_lookup_references(index_names, legal_slots, max(top_k, legal_slots.expected_doc_count))
        metadata_route_count = len(metadata_references)
        metadata_route_used = bool(metadata_references)

    if _can_short_circuit_metadata(legal_slots, metadata_references):
        references = metadata_references[:top_k]
        trace = {
            "question": question,
            "standalone_query": effective_query,
            "planned_queries": [effective_query],
            "rewrite_applied": effective_query != (question or "").strip(),
            "route_mode": RETRIEVAL_ROUTE_MODE,
            "route_counter": {"legal_metadata": 1},
            "per_query": [],
            "merge_latency_ms": 0.0,
            "constraints": constraints,
            "legal_query_slots": legal_slots.to_trace_dict(),
            "legal_metadata_route_used": True,
            "legal_metadata_route_count": metadata_route_count,
            "legal_metadata_short_circuit": True,
            "announcement_fallback_used": False,
            "announcement_fallback_latency_ms": 0.0,
            "fallback_used": False,
            "fallback_latency_ms": 0.0,
            "returned_count": len(references),
            "total_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        if ENABLE_RETRIEVAL_TRACE:
            logger.info("retrieval_trace=%s", trace)
        return {
            "question": question,
            "standalone_query": effective_query,
            "planned_queries": [effective_query],
            "references": references,
            "trace": trace,
        }

    planned_queries = build_diversified_queries(effective_query, profile=effective_profile) or [effective_query]
    company = extract_company_candidate(effective_query)
    per_query_limit = max(top_k * 3, 6)
    chunk_lists: list[list[dict[str, Any]]] = []
    query_traces: list[dict[str, Any]] = []
    query_profiles = {planned_query: classify_query_intent(planned_query) for planned_query in planned_queries}

    route_counter: dict[str, int] = {}

    for planned_query in planned_queries:
        single_started_at = time.perf_counter()
        single_chunks, route_used = _retrieve_single(index_names, planned_query, per_query_limit)
        route_counter[route_used] = route_counter.get(route_used, 0) + 1
        planned_profile = query_profiles[planned_query]
        filtered_chunks = filter_chunks_by_query_context(
            single_chunks,
            company,
            planned_query,
            profile=planned_profile,
        )
        chunk_lists.append(filtered_chunks[:per_query_limit])
        query_traces.append(
            {
                "query": planned_query,
                "route": {
                    "selected_route": route_used,
                    "numeric": planned_profile.is_numeric,
                    "deadline": getattr(planned_profile, "is_deadline_query", False),
                    "amount": getattr(planned_profile, "is_amount_query", False),
                    "table": planned_profile.is_table,
                    "risk": planned_profile.is_risk,
                    "announcement": planned_profile.prefers_announcement,
                    "actual_announcement": planned_profile.wants_actual_announcement,
                    "commentary": planned_profile.prefers_commentary,
                    "numeric_commentary": planned_profile.prefers_numeric_commentary,
                    "summary": planned_profile.prefers_summary,
                    "exact_match": planned_profile.needs_exact_match,
                },
                "latency_ms": round((time.perf_counter() - single_started_at) * 1000, 2),
                "raw_count": len(single_chunks),
                "filtered_count": len(filtered_chunks),
                "top_documents": list(
                    dict.fromkeys(
                        chunk.get("document_name", "")
                        for chunk in filtered_chunks
                        if chunk.get("document_name")
                    )
                )[:3],
            }
        )

    merge_started_at = time.perf_counter()
    merged = merge_ranked_chunks(
        chunk_lists,
        limit=max(top_k, 5),
        company=company,
        planned_query=effective_query,
        profile=effective_profile,
    )
    merge_latency_ms = round((time.perf_counter() - merge_started_at) * 1000, 2)
    references = _evidence_rank(merged, effective_profile, constraints, legal_slots)[:top_k]
    fallback_used = False
    announcement_fallback_used = False
    announcement_fallback_latency_ms = 0.0
    if metadata_references:
        references = _prepend_unique_documents(metadata_references, references, top_k)
    if _looks_like_actual_announcement_query(effective_query, effective_profile) and (
        not references or not _is_announcement_like_reference(references[0])
    ):
        announcement_fallback_started_at = time.perf_counter()
        announcement_references = _announcement_docname_fallback(index_names, effective_query, max(top_k, 5))
        announcement_fallback_latency_ms = round((time.perf_counter() - announcement_fallback_started_at) * 1000, 2)
        if announcement_references:
            references = _prepend_unique_references(announcement_references, references, top_k)
            announcement_fallback_used = True

    if not references and ENABLE_FALLBACK_ROUTE:
        fallback_used = True
        fallback_started_at = time.perf_counter()
        references = legacy_retrieve_content(index_names, effective_query, top_k=top_k)
        fallback_latency_ms = round((time.perf_counter() - fallback_started_at) * 1000, 2)
    else:
        fallback_latency_ms = 0.0

    # 针对公报期次查询的强制兜底：若当前结果未命中公报文件，则再走一次 docname 回退
    issue_keys = _extract_gazette_issue_keys(effective_query)
    if _is_gazette_query(effective_query):
        has_gazette = references and any(_looks_like_gazette_reference(item) for item in references)
        covered_keys = {
            key
            for key in issue_keys
            if any(key in str(item.get("document_name", "") or "") for item in references)
        }
        if (not has_gazette) or (issue_keys and len(covered_keys) < min(2, len(issue_keys))):
            gazette_fallback = _announcement_docname_fallback(index_names, effective_query, max(top_k, 8))
            if gazette_fallback:
                references = _prepend_unique_references(gazette_fallback, references, top_k)
                announcement_fallback_used = True

    if metadata_references:
        expected_docs = max(1, int(constraints.get("expected_doc_count", 1) or 1))
        if expected_docs >= 2:
            references = _prepend_unique_references(references, metadata_references, top_k)
        else:
            references = _prepend_unique_documents(metadata_references, references, top_k)

    expected_docs = max(1, int(constraints.get("expected_doc_count", 1) or 1))
    if expected_docs >= 2:
        references = _ensure_distinct_docs(references, required_docs=2, limit=top_k)

    trace = {
        "question": question,
        "standalone_query": effective_query,
        "planned_queries": planned_queries,
        "rewrite_applied": effective_query != (question or "").strip(),
        "route_mode": RETRIEVAL_ROUTE_MODE,
        "route_counter": route_counter,
        "per_query": query_traces,
        "merge_latency_ms": merge_latency_ms,
        "constraints": constraints,
        "legal_query_slots": legal_slots.to_trace_dict(),
        "legal_metadata_route_used": metadata_route_used,
        "legal_metadata_route_count": metadata_route_count,
        "legal_metadata_short_circuit": False,
        "announcement_fallback_used": announcement_fallback_used,
        "announcement_fallback_latency_ms": announcement_fallback_latency_ms,
        "fallback_used": fallback_used,
        "fallback_latency_ms": fallback_latency_ms,
        "returned_count": len(references),
        "total_latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }

    if ENABLE_RETRIEVAL_TRACE:
        logger.info("retrieval_trace=%s", trace)

    return {
        "question": question,
        "standalone_query": effective_query,
        "planned_queries": planned_queries,
        "references": references,
        "trace": trace,
    }


def retrieve_content(
    index_names: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    history_turns: list[dict[str, Any]] | None = None,
    standalone_query: str | None = None,
) -> list[dict[str, Any]]:
    return build_retrieval_context(
        index_names,
        question,
        top_k=top_k,
        history_turns=history_turns,
        standalone_query=standalone_query,
    )["references"]
