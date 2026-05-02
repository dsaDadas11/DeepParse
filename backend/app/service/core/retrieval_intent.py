from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from runtime_config import (
    INTENT_AMOUNT_WEIGHT_BIAS,
    INTENT_DEADLINE_WEIGHT_BIAS,
    LEGAL_TERM_DICT_PATH,
    LEGAL_TERM_NORMALIZATION_ENABLED,
)

LEGAL_INTENTS = {
    "article_locate": ("第", "条", "款", "项", "怎么规定", "规定是什么"),
    "procedure_extract": ("申请条件", "申请材料", "办理流程", "办理时限", "受理机关", "许可", "怎么办"),
    "penalty_elements": ("处罚", "构成要件", "适用条件", "情形", "违法"),
    "case_holding": ("案例要旨", "裁判要旨", "裁判结论", "指导性案例", "案号"),
    "case_reasoning": ("裁判理由", "争议焦点", "法院认为", "为什么"),
    "contract_clause": ("合同", "违约责任", "解除条件", "争议解决", "权利义务"),
    "doc_precise": ("原文", "全文", "是否包含", "精确定位", "第几条"),
    "version_effect": ("现行", "修订", "生效", "废止", "哪一版", "版本"),
    "comparison": ("比较", "对比", "区别", "哪个更", "差异"),
}

DOC_TYPE_HINTS = (
    "法律", "法典", "条例", "实施条例", "地方性法规", "司法解释", "指导性案例", "裁判文书", "合同范本", "办事指南", "公报",
)

AUTHORITY_HINTS = ("最高人民法院", "最高人民检察院", "国务院", "人民法院", "人民政府", "委员会", "厅", "局", "部")

ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千0-9]+(?:编|章|节|条|款|项)")
CASE_PATTERN = re.compile(r"指导性案例\s*第?\d+号|\(?\d{4}\)?[\u4e00-\u9fa5]{1,6}\d{2,8}号")
DATE_PATTERN = re.compile(r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?|\b(?:19|20)\d{6}\b")
TITLE_PATTERN = re.compile(r"《([^》]{2,80})》")
DOC_TITLE_CANDIDATE_PATTERN = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9·（）()、]{2,80}?(?:法典|法律|法|实施条例|条例|规定|办法|公报|合同|合同法|指南|申请表|申报表|案例))"
)
LEGAL_TABLE_TOKENS = ("申请表", "申报表", "表单", "目录", "清单")
LEGAL_CASE_HINTS = ("指导性案例", "案号", "裁判要旨", "裁判理由", "争议焦点")
LEGAL_DEADLINE_HINTS = (
    "期限", "时限", "几日", "几天", "多久", "工作日", "办理时限", "审查时限", "受理时限", "截止", "届满", "起算",
)
LEGAL_AMOUNT_HINTS = (
    "金额", "罚款", "罚金", "赔偿", "违约金", "数额", "标准", "上限", "下限", "元", "万元", "比例",
)

DEFAULT_TERM_NORMALIZATION_MAP: dict[str, tuple[str, ...]] = {
    "合同解除": ("解除合同",),
    "解除合同": ("合同解除",),
    "违约金": ("违约损害赔偿",),
    "违约损害赔偿": ("违约金",),
    "生效日期": ("生效时间", "施行日期"),
    "施行日期": ("生效日期",),
    "法条": ("条文", "法规条文"),
}


@dataclass(frozen=True)
class QueryIntentProfile:
    query: str
    normalized_query: str
    intent: str
    doc_type_terms: tuple[str, ...]
    authority_terms: tuple[str, ...]
    article_terms: tuple[str, ...]
    case_terms: tuple[str, ...]
    date_terms: tuple[str, ...]
    title_terms: tuple[str, ...]
    needs_exact_match: bool
    needs_abstain_guard: bool
    is_comparison: bool
    prefers_sparse: bool
    prefers_dense: bool
    is_deadline_query: bool
    is_amount_query: bool
    company: str
    brokers: tuple[str, ...]
    period_tokens: tuple[str, ...]
    metric_terms: tuple[str, ...]
    # compatibility fields required by existing rerank/retrieval path
    is_forecast: bool
    is_numeric: bool
    is_table: bool
    is_risk: bool
    is_commentary: bool
    is_announcement: bool
    prefers_summary: bool
    wants_actual_announcement: bool
    prefers_numeric_commentary: bool
    prefers_announcement: bool
    prefers_commentary: bool
    needs_diverse_documents: bool


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        value = value.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _load_term_normalization_map() -> dict[str, tuple[str, ...]]:
    mapping: dict[str, tuple[str, ...]] = dict(DEFAULT_TERM_NORMALIZATION_MAP)
    configured = Path(LEGAL_TERM_DICT_PATH).expanduser() if LEGAL_TERM_DICT_PATH else None
    if not configured:
        return mapping
    try:
        if not configured.exists() or not configured.is_file():
            return mapping
        payload = json.loads(configured.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return mapping
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, list):
                aliases = tuple(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str):
                aliases = (value.strip(),) if value.strip() else ()
            else:
                aliases = ()
            if aliases:
                mapping[key.strip()] = aliases
    except Exception:
        return mapping
    return mapping


TERM_NORMALIZATION_MAP = _load_term_normalization_map()


def normalize_query_terms(query: str) -> str:
    text = (query or "").strip()
    if not text or not LEGAL_TERM_NORMALIZATION_ENABLED:
        return text

    additions: list[str] = []
    for canonical, aliases in TERM_NORMALIZATION_MAP.items():
        terms = (canonical, *aliases)
        if any(term and term in text for term in terms):
            additions.extend(term for term in terms if term)

    if not additions:
        return text
    unique_additions = _unique(additions)
    return f"{text} {' '.join(unique_additions)}".strip()


def _detect_intent(text: str) -> str:
    if any(token in text for token in ("同时", "并列", "对比", "比较", "区别")):
        return "comparison"
    if any(token in text for token in ("版本", "版", "生效", "修订", "现行")):
        return "version_effect"
    if "指导性案例" in text or CASE_PATTERN.search(text):
        return "case_holding"
    if ARTICLE_PATTERN.search(text):
        return "article_locate"
    if any(token in text for token in ("办事指南", "申请条件", "申请材料", "办理流程", "办理时限")):
        return "procedure_extract"
    if "合同" in text and any(token in text for token in ("条款", "违约责任", "解除条件", "争议解决", "版", "版本")):
        return "contract_clause"
    for intent, hints in LEGAL_INTENTS.items():
        if any(h in text for h in hints):
            return intent
    return "doc_precise"


def classify_query_intent(query: str) -> QueryIntentProfile:
    raw_query = (query or "").strip()
    expanded_query = normalize_query_terms(raw_query)
    normalized_query = re.sub(r"\s+", "", expanded_query)

    intent = _detect_intent(expanded_query)
    doc_type_terms = _unique([term for term in DOC_TYPE_HINTS if term in expanded_query])
    authority_terms = _unique([term for term in AUTHORITY_HINTS if term in expanded_query])
    article_terms = _unique(ARTICLE_PATTERN.findall(expanded_query))
    case_terms = _unique(CASE_PATTERN.findall(expanded_query))
    date_terms = _unique(DATE_PATTERN.findall(expanded_query))

    titled = list(TITLE_PATTERN.findall(expanded_query))
    if not titled:
        title_candidates = DOC_TITLE_CANDIDATE_PATTERN.findall(expanded_query)
        noise_prefixes = (
            "请", "给", "我", "你", "帮", "返回", "定位", "检索", "提供", "回答", "并列", "同时", "不要", "只要", "答案"
        )
        for candidate in title_candidates:
            c = (candidate or "").strip("，。；：:、 ")
            if not c:
                continue
            if c.startswith(noise_prefixes):
                continue
            titled.append(c)
    title_terms = _unique(titled)
    company = ""
    brokers: tuple[str, ...] = ()
    period_tokens = date_terms

    table_metric_terms = [token for token in LEGAL_TABLE_TOKENS if token in expanded_query]
    metric_terms = _unique(list(article_terms) + list(case_terms) + list(title_terms) + table_metric_terms)

    is_deadline_query = any(token in expanded_query for token in LEGAL_DEADLINE_HINTS)
    is_amount_query = any(token in expanded_query for token in LEGAL_AMOUNT_HINTS)

    needs_exact_match = bool(article_terms or case_terms or date_terms or title_terms) or intent in {
        "article_locate",
        "doc_precise",
        "version_effect",
        "procedure_extract",
        "contract_clause",
    }
    is_comparison = intent == "comparison"
    needs_abstain_guard = any(token in expanded_query for token in ("有没有", "是否有", "是否包含", "找不到", "没有的话")) or intent in {
        "doc_precise",
        "version_effect",
    }

    is_table_query = any(token in expanded_query for token in LEGAL_TABLE_TOKENS)
    is_case_query = any(token in expanded_query for token in LEGAL_CASE_HINTS) or bool(case_terms)

    prefers_sparse = needs_exact_match or intent in {"procedure_extract", "contract_clause", "article_locate", "version_effect"}
    prefers_dense = intent in {"case_reasoning", "case_holding", "comparison", "penalty_elements"}

    if is_table_query or is_case_query or is_deadline_query or is_amount_query:
        needs_exact_match = True

    return QueryIntentProfile(
        query=raw_query,
        normalized_query=normalized_query,
        intent=intent,
        doc_type_terms=doc_type_terms,
        authority_terms=authority_terms,
        article_terms=article_terms,
        case_terms=case_terms,
        date_terms=date_terms,
        title_terms=title_terms,
        needs_exact_match=needs_exact_match,
        needs_abstain_guard=needs_abstain_guard,
        is_comparison=is_comparison,
        prefers_sparse=prefers_sparse,
        prefers_dense=prefers_dense,
        is_deadline_query=is_deadline_query,
        is_amount_query=is_amount_query,
        company=company,
        brokers=brokers,
        period_tokens=period_tokens,
        metric_terms=metric_terms,
        is_forecast=False,
        is_numeric=needs_exact_match or is_deadline_query or is_amount_query,
        is_table=(intent == "procedure_extract" or is_table_query),
        is_risk=(intent == "penalty_elements"),
        is_commentary=(intent in {"case_reasoning", "case_holding"}),
        is_announcement=(intent == "doc_precise" and "公报" in expanded_query),
        prefers_summary=is_table_query,
        wants_actual_announcement=("公报" in expanded_query and any(token in expanded_query for token in ("第", "期", "并列", "同时"))),
        prefers_numeric_commentary=False,
        prefers_announcement=(intent == "doc_precise") or ("公报" in expanded_query),
        prefers_commentary=(intent in {"case_reasoning", "case_holding", "comparison"}),
        needs_diverse_documents=is_comparison,
    )


def fusion_weights_for_profile(profile: QueryIntentProfile) -> tuple[float, float]:
    text_weight = 0.12
    if profile.prefers_sparse:
        text_weight += 0.12
    if profile.needs_exact_match:
        text_weight += 0.06
    if profile.is_deadline_query:
        text_weight += float(INTENT_DEADLINE_WEIGHT_BIAS)
    if profile.is_amount_query:
        text_weight += float(INTENT_AMOUNT_WEIGHT_BIAS)
    if profile.is_comparison:
        text_weight -= 0.02
    if profile.prefers_dense:
        text_weight -= 0.03
    text_weight = min(0.42, max(0.08, text_weight))
    return text_weight, 1.0 - text_weight


def rerank_vector_weight_for_profile(
    profile: QueryIntentProfile,
    default_vector_weight: float = 0.6,
) -> float:
    vector_weight = float(default_vector_weight)
    if profile.prefers_sparse:
        vector_weight -= 0.08
    if profile.needs_exact_match:
        vector_weight -= 0.04
    if profile.is_deadline_query:
        vector_weight -= min(0.08, float(INTENT_DEADLINE_WEIGHT_BIAS))
    if profile.is_amount_query:
        vector_weight -= min(0.08, float(INTENT_AMOUNT_WEIGHT_BIAS))
    if profile.prefers_dense:
        vector_weight += 0.06
    if profile.is_comparison:
        vector_weight += 0.04
    return min(0.82, max(0.30, vector_weight))
