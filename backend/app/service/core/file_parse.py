import datetime
import json
import os
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List

import xxhash

from service.core.document_metadata import parse_document_metadata
from service.core.rag.nlp import rag_tokenizer
from service.core.rag.app.naive import chunk
from service.core.rag.nlp.model import EmbeddingGenerationError, generate_embedding
from service.core.rag.utils import num_tokens_from_string
from service.core.rag.utils.es_conn import ESConnection
from service.core.rag_config import (
    DEFAULT_CHUNK_DELIMITER,
    DEFAULT_CHUNK_TOKEN_NUM,
    DEFAULT_LAYOUT_RECOGNIZE,
    PDF_PLAIN_TEXT_SIZE_THRESHOLD_BYTES,
)
from utils import logger


def dummy(prog=None, msg=""):
    return None


LEGAL_SECTION_PATTERN = re.compile(r"(第[一二三四五六七八九十百千0-9]+(?:编|章|节|条|款|项))")
LEGAL_CASE_PATTERN = re.compile(r"(指导性案例\s*第?\d+号|\(?\d{4}\)?[\u4e00-\u9fa5]{1,6}\d{2,8}号|案号[:：]?\s*[^\s，。；]{4,40})")
PROCEDURE_ANCHOR_PATTERN = re.compile(r"(申请条件|申请材料|办理流程|审查时限|受理机关|办理时限)")
CONTRACT_ANCHOR_PATTERN = re.compile(r"(合同目的|权利义务|违约责任|解除条件|争议解决|适用法律)")
MAX_EMBEDDING_TEXT_TOKENS = 6000
HTML_ROW_PATTERN = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
HTML_CELL_PATTERN = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
TABLE_SECTION_HINTS = (
    "关键指标",
    "主要会计数据",
    "主要财务指标",
    "主要财务数据",
    "财务摘要",
    "营业收入构成及变动情况",
)
TABLE_DENSE_REPORT_HINTS = ("表格密集版", "摘要")
TABLE_ROW_BOUNDARY_HINTS = (
    "营业收入",
    "营业总收入",
    "营收",
    "归属于上市公司股东的净利润",
    "归属于母公司股东的净利润",
    "归属于本行股东的净利润",
    "归母净利润",
    "归母净利",
    "净利润",
    "扣非净利润",
    "利息净收入",
    "非利息净收入",
    "经营活动现金流量净额",
    "经营活动产生的现金流量净额",
    "研发投入合计",
    "每股现金分红",
    "现金红利",
    "净息差",
    "不良贷款率",
    "拨备覆盖率",
    "核心一级资本充足率",
    "一级资本充足率",
    "资本充足率",
    "毛利率",
    "归母净利率",
    "成本收入比",
    "平均总资产收益率",
    "加权平均净资产收益率",
)
TABLE_ROW_MAX_CHARS = 240
TABLE_HEADER_MAX_CHARS = 160
FORWARD_TABLE_HEADER_PATTERN = re.compile(
    r"(?:(?:货币单位|单位)[:：]?[^\n。；]{0,40})?(?:项\s*目)?[^\n。；]{0,24}"
    r"(?:19\d{2}|20\d{2})\s*年?[^\n。；]{0,80}(?:19\d{2}|20\d{2})\s*年?[^\n。；]{0,80}"
)


def _format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _parse_decimal(raw_value: str) -> Decimal | None:
    try:
        return Decimal(raw_value.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def clean_table_text(text: str) -> str:
    normalized = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", text or "", flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_numeric_tokens(text: str) -> list[str]:
    values: list[str] = []
    for token in re.findall(r"[+-]?\d[\d,]*(?:\.\d+)?", text or ""):
        compact = token.replace(",", "")
        if re.fullmatch(r"(?:19\d{2}|20\d{2})", compact):
            continue
        values.append(compact)
    return values


def extract_year_tokens(text: str) -> list[str]:
    years = re.findall(r"(?<!\d)(?:19\d{2}|20\d{2})(?=\s*年?)", text or "")
    seen: set[str] = set()
    ordered: list[str] = []
    for year in years:
        if year in seen:
            continue
        seen.add(year)
        ordered.append(year)
    return ordered


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def iter_html_table_rows(content: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_row in HTML_ROW_PATTERN.findall(content or ""):
        cells = [clean_table_text(cell) for cell in HTML_CELL_PATTERN.findall(raw_row)]
        cleaned_cells = [cell for cell in cells if cell]
        if cleaned_cells:
            rows.append(cleaned_cells)
    return rows


def is_table_dense_chunk(content: str, *, report_type: str = "", document_name: str = "") -> bool:
    cleaned = clean_table_text(content)
    if not cleaned:
        return False
    if "<table" in (content or "").lower():
        return True

    numeric_count = len(extract_numeric_tokens(cleaned))
    year_count = len(extract_year_tokens(cleaned))
    metadata_text = clean_table_text(f"{report_type} {document_name}")
    if any(hint in metadata_text for hint in TABLE_DENSE_REPORT_HINTS):
        return numeric_count >= 3 and (year_count >= 1 or any(hint in cleaned for hint in TABLE_SECTION_HINTS))
    return numeric_count >= 4 and year_count >= 1 and any(hint in cleaned for hint in TABLE_SECTION_HINTS)


def extract_dense_table_header(cleaned: str, first_row_index: int) -> str:
    if first_row_index <= 0:
        prefix = cleaned[:TABLE_HEADER_MAX_CHARS]
    else:
        prefix = cleaned[max(0, first_row_index - TABLE_HEADER_MAX_CHARS):first_row_index]
    prefix = clean_table_text(prefix)
    if not prefix:
        return ""
    for hint in TABLE_SECTION_HINTS:
        hint_index = prefix.rfind(hint)
        if hint_index >= 0:
            prefix = prefix[hint_index:]
            break
    return prefix[:TABLE_HEADER_MAX_CHARS]


def extract_forward_table_header(cleaned: str) -> str:
    match = FORWARD_TABLE_HEADER_PATTERN.search(cleaned)
    if not match:
        return ""
    return clean_table_text(match.group(0))[:TABLE_HEADER_MAX_CHARS]


def extract_dense_table_rows(content: str) -> tuple[list[str], list[str]]:
    rows = iter_html_table_rows(content)
    if rows:
        header_rows = [
            " ".join(cell for cell in row if cell).strip()
            for row in rows[:2]
            if row and (len(extract_year_tokens(" ".join(row))) >= 1 or any(hint in " ".join(row) for hint in TABLE_SECTION_HINTS))
        ]
        row_texts = [
            " ".join(cell for cell in row if cell).strip()[:TABLE_ROW_MAX_CHARS]
            for row in rows
            if len(extract_numeric_tokens(" ".join(row))) >= 1
        ]
        return unique_preserve_order(header_rows[:2]), unique_preserve_order(row_texts[:16])

    cleaned = clean_table_text(content)
    if not cleaned:
        return [], []

    matches: list[tuple[int, str]] = []
    for hint in TABLE_ROW_BOUNDARY_HINTS:
        start = 0
        while True:
            index = cleaned.find(hint, start)
            if index < 0:
                break
            matches.append((index, hint))
            start = index + len(hint)
    matches.sort(key=lambda item: item[0])

    row_starts: list[int] = []
    for index, _ in matches:
        if row_starts and index - row_starts[-1] < 6:
            continue
        row_starts.append(index)

    if not row_starts:
        return [], []

    header_text = extract_dense_table_header(cleaned, row_starts[0])
    if len(extract_year_tokens(header_text)) < 1:
        forward_header = extract_forward_table_header(cleaned)
        if forward_header:
            header_text = forward_header
    row_texts: list[str] = []
    for i, row_start in enumerate(row_starts):
        row_end = row_starts[i + 1] if i + 1 < len(row_starts) else len(cleaned)
        row_end = min(row_end, row_start + TABLE_ROW_MAX_CHARS)
        segment = clean_table_text(cleaned[row_start:row_end])
        if len(extract_numeric_tokens(segment)) < 1:
            continue
        row_texts.append(segment[:TABLE_ROW_MAX_CHARS])

    headers = [header_text] if header_text else []
    return unique_preserve_order(headers[:2]), unique_preserve_order(row_texts[:16])


def build_table_annotations(content: str, *, report_type: str = "", document_name: str = "") -> dict[str, Any]:
    if not is_table_dense_chunk(content, report_type=report_type, document_name=document_name):
        return {"table_dense_int": 0, "table_headers_kwd": [], "table_rows_kwd": []}

    table_headers, table_rows = extract_dense_table_rows(content)
    if not table_rows:
        return {"table_dense_int": 0, "table_headers_kwd": [], "table_rows_kwd": []}

    return {
        "table_dense_int": 1,
        "table_headers_kwd": table_headers[:2],
        "table_rows_kwd": table_rows[:16],
    }


def append_table_retrieval_context(enriched_content: str, table_headers: list[str], table_rows: list[str]) -> str:
    additions = [header for header in table_headers[:2] if header]
    additions.extend(row for row in table_rows[:6] if row)
    additions = unique_preserve_order(additions)
    if not additions:
        return enriched_content
    return f"{enriched_content}\n{' '.join(additions)}"


def _extract_anchors_for_enrichment(text: str) -> list[str]:
    anchors: list[str] = []
    for pattern in (LEGAL_SECTION_PATTERN, LEGAL_CASE_PATTERN, PROCEDURE_ANCHOR_PATTERN, CONTRACT_ANCHOR_PATTERN):
        for item in pattern.findall(text):
            anchor = str(item).strip()
            if not anchor or anchor in anchors:
                continue
            anchors.append(anchor)
            if len(anchors) >= 16:
                return anchors
    return anchors


def enrich_chunk_text(content: str, metadata) -> str:
    normalized = re.sub(r"\s+", " ", content).strip()
    if not normalized:
        return content

    extras: list[str] = []
    header_parts = [
        part
        for part in (
            metadata.company,
            metadata.doc_type,
            metadata.authority,
            metadata.legal_domain,
            metadata.effective_or_revision_date,
            metadata.version_scope,
        )
        if part
    ]
    if header_parts:
        extras.append(" ".join(header_parts))

    anchors = _extract_anchors_for_enrichment(normalized)
    if anchors:
        extras.append(" ".join(anchors))

    if getattr(metadata, "keywords", None):
        keyword_boost = " ".join(list(metadata.keywords)[:20])
        if keyword_boost:
            extras.append(keyword_boost)

    if not extras:
        return normalized
    return f"{normalized}\n{' '.join(unique_preserve_order(extras))}"


def build_metadata_preview(items: List[Dict[str, Any]], max_fragments: int = 8, max_chars: int = 2400) -> str:
    preview_parts: list[str] = []
    total_chars = 0

    for item in items:
        raw_content = str(item.get("content_with_weight", "")).strip()
        if not raw_content:
            continue
        normalized = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", raw_content, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            continue

        snippet = normalized[:320]
        preview_parts.append(snippet)
        total_chars += len(snippet)
        if len(preview_parts) >= max_fragments or total_chars >= max_chars:
            break

    return "\n".join(preview_parts)


def clip_text_for_embedding(text: str, max_tokens: int = MAX_EMBEDDING_TEXT_TOKENS) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return normalized

    token_count = num_tokens_from_string(normalized)
    if token_count <= max_tokens:
        return normalized

    segments = [segment for segment in re.split(r"(?<=[。！？!?；;\n])", normalized) if segment and segment.strip()]
    if not segments:
        segments = [normalized]

    clipped_parts: list[str] = []
    clipped_tokens = 0
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        segment_tokens = num_tokens_from_string(segment)
        if clipped_parts and clipped_tokens + segment_tokens > max_tokens:
            break
        if not clipped_parts and segment_tokens > max_tokens:
            ratio = max_tokens / max(segment_tokens, 1)
            clipped_length = max(256, int(len(segment) * ratio * 0.95))
            segment = segment[:clipped_length].rstrip()
            segment_tokens = num_tokens_from_string(segment)
        clipped_parts.append(segment)
        clipped_tokens += segment_tokens

    clipped_text = "\n".join(clipped_parts).strip() or normalized[:4096]
    while num_tokens_from_string(clipped_text) > max_tokens and len(clipped_text) > 256:
        clipped_text = clipped_text[: int(len(clipped_text) * 0.9)].rstrip()

    logger.warning(
        "Clipped embedding text from %s tokens to %s tokens.",
        token_count,
        num_tokens_from_string(clipped_text),
    )
    return clipped_text


def build_parser_config(file_path: str) -> Dict[str, Any]:
    parser_config: Dict[str, Any] = {
        "chunk_token_num": DEFAULT_CHUNK_TOKEN_NUM,
        "delimiter": "\n!?。；！？",
        "layout_recognize": DEFAULT_LAYOUT_RECOGNIZE,
    }
    parser_config["delimiter"] = DEFAULT_CHUNK_DELIMITER

    if not file_path.lower().endswith(".pdf"):
        return parser_config

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
    metadata = parse_document_metadata(file_name)

    is_large_pdf = file_size >= PDF_PLAIN_TEXT_SIZE_THRESHOLD_BYTES
    is_statute_like = bool(metadata.doc_type in {"法律", "条例", "地方性法规", "司法解释"})
    is_case_like = bool(metadata.doc_type in {"指导性案例", "裁判文书"})
    is_contract_like = bool(metadata.doc_type == "合同范本")
    is_procedure_like = bool(metadata.doc_type in {"办事指南", "公报"})

    if is_statute_like:
        parser_config["chunk_token_num"] = max(parser_config["chunk_token_num"], 196)
    if is_case_like:
        parser_config["chunk_token_num"] = max(parser_config["chunk_token_num"], 220)
    if is_contract_like:
        parser_config["chunk_token_num"] = max(parser_config["chunk_token_num"], 200)
    if is_procedure_like:
        parser_config["chunk_token_num"] = max(parser_config["chunk_token_num"], 180)

    if is_large_pdf or re.search(r"第[一二三四五六七八九十百千0-9]+条|指导性案例|案号|申请材料|办理流程|违约责任", file_name):
        parser_config["chunk_token_num"] = max(parser_config["chunk_token_num"], 224)

    if is_large_pdf or is_statute_like or is_case_like or is_contract_like or is_procedure_like:
        parser_config["layout_recognize"] = "Plain Text"

    return parser_config


def parse(file_path: str):
    parser_config = build_parser_config(file_path)
    items = chunk(file_path, callback=dummy, parser_config=parser_config)
    if items or not file_path.lower().endswith(".pdf"):
        return items

    if parser_config.get("layout_recognize") != "Plain Text":
        return items

    fallback_config = dict(parser_config)
    fallback_config["layout_recognize"] = "DeepDOC"
    return chunk(file_path, callback=dummy, parser_config=fallback_config)


def batch_generate_embeddings(texts: List[str], batch_size: int = 10) -> List[List[float]]:
    if not texts:
        return []
    embeddings = generate_embedding(texts, max_batch_size=batch_size)
    if embeddings is None:
        raise RuntimeError("Embedding service returned no embeddings.")
    return embeddings


def build_retrieval_tokens(text: str) -> tuple[str, str]:
    normalized = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", text or "")
    content_ltks = rag_tokenizer.tokenize(normalized)
    content_sm_ltks = rag_tokenizer.fine_grained_tokenize(content_ltks)
    return content_ltks, content_sm_ltks


def merge_keyword_values(*values: Any) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        items = value if isinstance(value, (list, tuple, set)) else [value]
        for item in items:
            normalized = str(item).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged


def build_chunk_id(
    *,
    index_name: str,
    file_name: str,
    ordinal: int,
    content: str,
    page_num_int: list[int],
    position_int: list[Any],
) -> str:
    payload = json.dumps(
        {
            "index_name": index_name,
            "file_name": file_name,
            "ordinal": ordinal,
            "page_num_int": page_num_int,
            "position_int": position_int,
            "content": content,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return xxhash.xxh64(payload.encode("utf-8")).hexdigest()


def process_items(items: List[Dict[str, Any]], file_name: str, index_name: str) -> List[Dict[str, Any]]:
    try:
        metadata_preview = build_metadata_preview(items)
        metadata = parse_document_metadata(file_name, preview_text=metadata_preview)
        doc_id = xxhash.xxh64(f"{index_name}:{file_name}".encode("utf-8")).hexdigest()
        prepared_items: list[dict[str, Any]] = []
        texts: list[str] = []

        for ordinal, item in enumerate(items, start=1):
            original_content = str(item.get("content_with_weight", "")).strip()
            if not original_content:
                continue

            table_annotations = build_table_annotations(
                original_content,
                report_type=metadata.report_type or "",
                document_name=file_name,
            )
            enriched_content = enrich_chunk_text(original_content, metadata)
            if table_annotations["table_dense_int"]:
                enriched_content = append_table_retrieval_context(
                    enriched_content,
                    list(table_annotations["table_headers_kwd"]),
                    list(table_annotations["table_rows_kwd"]),
                )
            content_ltks, content_sm_ltks = build_retrieval_tokens(enriched_content)
            metadata_keywords = list(metadata.keywords)
            important_kwd = merge_keyword_values(metadata_keywords, item.get("important_kwd"))
            question_kwd = merge_keyword_values(metadata_keywords, item.get("question_kwd"))
            question_tks = rag_tokenizer.tokenize(" ".join(question_kwd))
            raw_title_tokens = str(item.get("title_tks", "")).split()
            merged_title_values = merge_keyword_values(
                metadata.company,
                metadata.report_period,
                metadata.report_type,
                metadata.source,
                raw_title_tokens,
            )
            title_tks = rag_tokenizer.tokenize(" ".join(merged_title_values))
            page_num_int = [int(page) for page in (item.get("page_num_int") or [])]
            position_int = list(item.get("position_int") or [])
            top_int = [int(top) for top in (item.get("top_int") or [])]

            prepared_items.append(
                {
                    "ordinal": ordinal,
                    "original_content": original_content,
                    "content_ltks": content_ltks,
                    "content_sm_ltks": content_sm_ltks,
                    "important_kwd": important_kwd,
                    "important_tks": rag_tokenizer.tokenize(" ".join(important_kwd)),
                    "question_kwd": question_kwd,
                    "question_tks": question_tks,
                    "page_num_int": page_num_int,
                    "position_int": position_int,
                    "top_int": top_int,
                    "title_tks": title_tks,
                    "docnm_kwd": item.get("docnm_kwd", file_name),
                    "img_id": item.get("img_id", ""),
                    "knowledge_graph_kwd": list(item.get("knowledge_graph_kwd") or []),
                    "company_kwd": metadata.company or "",
                    "report_period_kwd": metadata.report_period or "",
                    "report_type_kwd": metadata.report_type or "",
                    "source_kwd": metadata.source or "",
                    "doc_type_kwd": metadata.doc_type or "",
                    "authority_kwd": metadata.authority or "",
                    "legal_domain_kwd": metadata.legal_domain or "",
                    "effective_or_revision_date_kwd": metadata.effective_or_revision_date or "",
                    "version_scope_kwd": metadata.version_scope or "",
                    "article_anchors_kwd": list(metadata.article_anchors),
                    "case_anchors_kwd": list(metadata.case_anchors),
                    "procedure_anchors_kwd": list(metadata.procedure_anchors),
                    "contract_anchors_kwd": list(metadata.contract_anchors),
                    "table_dense_int": int(table_annotations["table_dense_int"]),
                    "table_headers_kwd": list(table_annotations["table_headers_kwd"]),
                    "table_rows_kwd": list(table_annotations["table_rows_kwd"]),
                }
            )
            texts.append(clip_text_for_embedding(enriched_content))

        try:
            embeddings = batch_generate_embeddings(texts)
        except EmbeddingGenerationError as exc:
            failed_indices = sorted(set(getattr(exc, "failed_indices", [])))
            failed_ordinals = [
                prepared_items[index]["ordinal"]
                for index in failed_indices
                if 0 <= index < len(prepared_items)
            ]
            sample = ", ".join(str(ordinal) for ordinal in failed_ordinals[:5]) or "unknown"
            logger.error(
                "Embedding generation failed for %s; failed chunk ordinals=%s",
                file_name,
                failed_ordinals or failed_indices,
            )
            raise RuntimeError(
                f"Embedding generation failed for {len(failed_ordinals) or len(failed_indices)} chunk(s) in {file_name}; "
                f"sample ordinals: {sample}."
            ) from exc
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedding count mismatch for {file_name}: expected {len(texts)}, got {len(embeddings)}."
            )

        missing_ordinals = [
            prepared_item["ordinal"]
            for prepared_item, embedding in zip(prepared_items, embeddings)
            if not embedding
        ]
        if missing_ordinals:
            sample = ", ".join(str(ordinal) for ordinal in missing_ordinals[:5])
            raise RuntimeError(
                f"Embedding generation failed for {len(missing_ordinals)} chunk(s) in {file_name}; "
                f"sample ordinals: {sample}."
            )

        results = []
        for item, embedding in zip(prepared_items, embeddings):
            chunk_id = build_chunk_id(
                index_name=index_name,
                file_name=file_name,
                ordinal=item["ordinal"],
                content=item["original_content"],
                page_num_int=item["page_num_int"],
                position_int=item["position_int"],
            )
            now = datetime.datetime.now()

            document = {
                "id": chunk_id,
                "content_ltks": item["content_ltks"],
                "content_with_weight": item["original_content"],
                "content_sm_ltks": item["content_sm_ltks"],
                "important_kwd": item["important_kwd"],
                "important_tks": item["important_tks"],
                "question_kwd": item["question_kwd"],
                "question_tks": item["question_tks"],
                "create_time": str(now).replace("T", " ")[:19],
                "create_timestamp_flt": now.timestamp(),
                "kb_id": index_name,
                "docnm_kwd": item["docnm_kwd"],
                "title_tks": item["title_tks"],
                "doc_id": doc_id,
                "docnm": file_name,
                "company_kwd": item["company_kwd"],
                "report_period_kwd": item["report_period_kwd"],
                "report_type_kwd": item["report_type_kwd"],
                "source_kwd": item["source_kwd"],
                "doc_type_kwd": item["doc_type_kwd"],
                "authority_kwd": item["authority_kwd"],
                "legal_domain_kwd": item["legal_domain_kwd"],
                "effective_or_revision_date_kwd": item["effective_or_revision_date_kwd"],
                "version_scope_kwd": item["version_scope_kwd"],
                "article_anchors_kwd": item["article_anchors_kwd"],
                "case_anchors_kwd": item["case_anchors_kwd"],
                "procedure_anchors_kwd": item["procedure_anchors_kwd"],
                "contract_anchors_kwd": item["contract_anchors_kwd"],
                "table_dense_int": item["table_dense_int"],
                "table_headers_kwd": item["table_headers_kwd"],
                "table_rows_kwd": item["table_rows_kwd"],
                "page_num_int": item["page_num_int"],
                "position_int": item["position_int"],
                "top_int": item["top_int"],
                "img_id": item["img_id"],
                "knowledge_graph_kwd": item["knowledge_graph_kwd"],
                "available_int": 1,
            }
            document[f"q_{len(embedding)}_vec"] = embedding
            results.append(document)

        return results
    except Exception as exc:
        logger.exception("process_items failed for %s", file_name)
        raise RuntimeError(f"Failed to prepare index documents for {file_name}: {exc}") from exc


def execute_insert_process(file_path: str, file_name: str, index_name: str):
    documents = parse(file_path)
    if not documents:
        raise RuntimeError(f"No documents were parsed from {file_path}.")

    processed_documents = process_items(documents, file_name, index_name)
    if not processed_documents:
        raise RuntimeError(f"Failed to build indexable chunks for {file_name}.")

    try:
        es_connection = ESConnection()
        es_connection.insert(documents=processed_documents, indexName=index_name)
        print(f"Successfully inserted {len(processed_documents)} documents into ES")
        return len(processed_documents)
    except Exception as exc:
        raise RuntimeError(f"Failed to insert documents into ES: {exc}") from exc
