from __future__ import annotations

import os
import re
from dataclasses import dataclass

from service.core.rag.nlp import rag_tokenizer

DOC_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("法律", ("法典", "法律", "法", "法案")),
    ("条例", ("条例", "实施条例", "办法", "细则", "规定")),
    ("地方性法规", ("地方", "省", "市", "自治区", "自治州", "条例")),
    ("司法解释", ("司法解释", "解释", "批复", "答复", "意见")),
    ("指导性案例", ("指导性案例", "典型案例", "参考案例")),
    ("裁判文书", ("判决书", "裁定书", "调解书", "赔偿案", "纠纷案", "刑事案", "知识产权案", "案例")),
    ("合同范本", ("合同", "协议", "示范文本", "范本")),
    ("办事指南", ("办事指南", "办事", "申请", "许可", "审批", "申报", "服务指南", "目录", "编码规则", "申报表")),
    ("公报", ("公报", "通告", "公告")),
)

LEGAL_DOMAIN_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("民法", ("民法", "民事", "婚姻", "继承", "人格权", "物权", "侵权")),
    ("刑法", ("刑法", "刑事", "犯罪", "量刑", "刑罚")),
    ("行政法", ("行政", "许可", "复议", "处罚", "政务")),
    ("知识产权", ("知识产权", "专利", "商标", "著作权", "版权")),
    ("税法", ("税", "税收", "纳税", "增值税", "企业所得税")),
    ("劳动法", ("劳动", "社保", "工伤", "劳动合同", "工资")),
    ("网络安全", ("网络", "数据", "个人信息", "隐私", "信息安全", "网络安全")),
)

AUTHORITY_PATTERN = re.compile(
    r"((?:最高人民法院|最高人民检察院|国务院|全国人民代表大会常务委员会|全国人民代表大会|"
    r"[\u4e00-\u9fa5]{2,18}(?:人民法院|人民检察院|人民政府|委员会|厅|局|部)))"
)
DATE_PATTERN = re.compile(r"((?:19|20)\d{2}年\d{1,2}月\d{1,2}日|(?:19|20)\d{2}年\d{1,2}月|(?:19|20)\d{2}年)")
ARTICLE_ANCHOR_PATTERN = re.compile(r"(第[一二三四五六七八九十百千0-9]+(?:编|章|节|条|款|项))")
CASE_ANCHOR_PATTERN = re.compile(
    r"((?:\(?\d{4}\)?[\u4e00-\u9fa5]{1,6}\d{2,8}号)|指导性案例\s*第?\d+号|案号[:：]?\s*[^\s，。；]{4,40}|案由[:：]?\s*[^\n，。；]{2,40})"
)
PROCEDURE_ANCHOR_PATTERN = re.compile(r"(申请条件|申请材料|办理流程|审查时限|受理机关|办理时限|办理地点)")
CONTRACT_ANCHOR_PATTERN = re.compile(r"(合同目的|权利义务|违约责任|解除条件|争议解决|适用法律|合同期限)")


@dataclass(frozen=True)
class DocumentMetadata:
    file_name: str
    base_name: str
    company: str | None
    report_period: str | None
    report_type: str | None
    source: str | None
    doc_type: str | None
    authority: str | None
    legal_domain: str | None
    effective_or_revision_date: str | None
    version_scope: str | None
    article_anchors: tuple[str, ...]
    case_anchors: tuple[str, ...]
    procedure_anchors: tuple[str, ...]
    contract_anchors: tuple[str, ...]
    keywords: tuple[str, ...]


def _normalize_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in rag_tokenizer.tokenize(value).split():
        normalized = token.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(normalized)
    return tokens


def _push_keyword(target: list[str], seen: set[str], value: str | None) -> None:
    if not value:
        return
    cleaned = value.strip()
    if not cleaned:
        return
    if cleaned not in seen:
        target.append(cleaned)
        seen.add(cleaned)
    for token in _normalize_tokens(cleaned):
        if token not in seen:
            target.append(token)
            seen.add(token)


def _push_many_keywords(target: list[str], seen: set[str], values: tuple[str, ...]) -> None:
    for value in values:
        _push_keyword(target, seen, value)


def _clean_preview_text(preview_text: str | None) -> str:
    if not preview_text:
        return ""
    preview = re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", preview_text, flags=re.IGNORECASE)
    preview = re.sub(r"[ \t]+", " ", preview)
    return preview.strip()


def _extract_doc_type(text: str) -> str | None:
    for doc_type, hints in DOC_TYPE_PATTERNS:
        if all(hint in text for hint in ("地方", "条例")) and doc_type == "地方性法规":
            return doc_type
        if any(hint in text for hint in hints):
            return doc_type
    return None


def _extract_authority(text: str) -> str | None:
    match = AUTHORITY_PATTERN.search(text)
    return match.group(1).strip() if match else None


def _extract_legal_domain(text: str) -> str | None:
    for domain, hints in LEGAL_DOMAIN_HINTS:
        if any(hint in text for hint in hints):
            return domain
    return None


def _extract_date(text: str) -> str | None:
    all_dates = DATE_PATTERN.findall(text)
    if not all_dates:
        return None
    return all_dates[-1]


def _extract_version_scope(text: str) -> str | None:
    if any(token in text for token in ("现行", "现行有效", "最新")):
        return "现行"
    if "修订" in text:
        return "修订版"
    match = re.search(r"((?:19|20)\d{2}年版)", text)
    if match:
        return match.group(1)
    return None


def _extract_anchors(pattern: re.Pattern[str], text: str, limit: int = 24) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for item in pattern.findall(text):
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        found.append(normalized)
        if len(found) >= limit:
            break
    return tuple(found)


def _primary_subject(base_name: str, doc_type: str | None) -> str | None:
    stem = base_name.replace("_", " ").strip()
    if not stem:
        return None
    if doc_type and doc_type in stem:
        stem = stem.replace(doc_type, "").strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem[:48] if stem else None


def parse_document_metadata(file_name: str, preview_text: str | None = None) -> DocumentMetadata:
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    preview = _clean_preview_text(preview_text)
    merged_text = f"{base_name} {preview}".strip()

    doc_type = _extract_doc_type(merged_text)
    authority = _extract_authority(merged_text)
    legal_domain = _extract_legal_domain(merged_text)
    effective_or_revision_date = _extract_date(merged_text)
    version_scope = _extract_version_scope(merged_text)
    article_anchors = _extract_anchors(ARTICLE_ANCHOR_PATTERN, merged_text)
    case_anchors = _extract_anchors(CASE_ANCHOR_PATTERN, merged_text)
    procedure_anchors = _extract_anchors(PROCEDURE_ANCHOR_PATTERN, merged_text)
    contract_anchors = _extract_anchors(CONTRACT_ANCHOR_PATTERN, merged_text)

    primary_subject = _primary_subject(base_name, doc_type)

    keywords: list[str] = []
    seen_keywords: set[str] = set()
    _push_keyword(keywords, seen_keywords, primary_subject)
    _push_keyword(keywords, seen_keywords, doc_type)
    _push_keyword(keywords, seen_keywords, authority)
    _push_keyword(keywords, seen_keywords, legal_domain)
    _push_keyword(keywords, seen_keywords, effective_or_revision_date)
    _push_keyword(keywords, seen_keywords, version_scope)
    _push_many_keywords(keywords, seen_keywords, article_anchors)
    _push_many_keywords(keywords, seen_keywords, case_anchors)
    _push_many_keywords(keywords, seen_keywords, procedure_anchors)
    _push_many_keywords(keywords, seen_keywords, contract_anchors)

    # 为兼容既有字段命名，沿用 company/report_period/report_type/source，但语义已法律化。
    return DocumentMetadata(
        file_name=os.path.basename(file_name),
        base_name=base_name,
        company=primary_subject,
        report_period=effective_or_revision_date or version_scope,
        report_type=doc_type,
        source=authority,
        doc_type=doc_type,
        authority=authority,
        legal_domain=legal_domain,
        effective_or_revision_date=effective_or_revision_date,
        version_scope=version_scope,
        article_anchors=article_anchors,
        case_anchors=case_anchors,
        procedure_anchors=procedure_anchors,
        contract_anchors=contract_anchors,
        keywords=tuple(keywords),
    )
