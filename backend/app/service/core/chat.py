import json
import os
import re
import time

from fastapi import HTTPException
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from sqlalchemy import text
from runtime_config import (
    CHAT_MODEL,
    CONFLICT_ABSTAIN_ENABLED,
    GENERATION_API_KEY,
    GENERATION_BASE_URL,
    RECOMMENDATION_MODEL,
    SESSION_NAME_MODEL,
    STRICT_CITATION_BINDING,
)
from service.core.answering_rules import citation_key_for_chunk, maybe_answer_by_rule
from service.core.conversation import format_history_for_prompt
from service.core.retrieval_intent import classify_query_intent
from utils import logger
from utils.database import get_db

NUMERIC_QUERY_HINTS = (
    "第",
    "条",
    "款",
    "项",
    "申请条件",
    "申请材料",
    "办理流程",
    "办理时限",
    "违约责任",
    "解除条件",
    "争议解决",
    "构成要件",
    "处罚",
    "生效",
    "修订",
    "案号",
    "版本日期",
)

ANALYSIS_QUERY_HINTS = (
    "裁判理由",
    "案例要旨",
    "争议焦点",
    "适用边界",
    "法条适用",
    "适用条件",
    "比较",
    "对比",
    "区别",
    "地方",
    "国家",
    "版本",
    "司法解释",
    "指导案例",
)

CITATION_PATTERN = re.compile(r"##(\d+)\$\$")
RULE_CITATION_PATTERN = re.compile(r"##ref_([A-Za-z0-9_-]+)\$\$")
ABSTAIN_MARKERS = ("无法回答", "无法确定", "未找到", "缺少直接证据", "无法支持该请求")
MAX_PROMPT_REFERENCES = 3
MAX_REFERENCE_CHARS = 420
MAX_PROMPT_TOTAL_CHARS = 2500
DEFAULT_CHAT_COMPLETION_TIMEOUT_SECONDS = 120

ROUTE_MODE_RULE = "rule"
ROUTE_MODE_MODEL = "model"
ROUTE_MODE_NONE = "none"

HIGH_CERTAINTY_HINTS = (
    "文件名",
    "表格文件名",
    "完整表名",
    "表名",
    "标题",
    "关键词",
    "条文编号",
    "版本",
    "版次",
    "版本号",
    "版本数字",
    "哪份",
    "案由关键词",
    "纠纷性质",
    "责任类型",
    "罪名方向",
    "发布单位",
    "原文文件",
    "文件",
    "文档名",
    "同时给出",
    "同时返回",
    "同时出现",
    "并列返回",
)
SEMANTIC_GENERATION_HINTS = (
    "解释",
    "说明",
    "含义",
    "理解",
    "为什么",
    "理由",
    "争议焦点",
    "法院认为",
    "裁判要旨",
    "案例要旨",
    "构成要件",
    "适用条件",
    "适用边界",
    "怎么规定",
    "如何适用",
    "归纳",
    "总结",
    "提炼",
    "比较",
    "对比",
    "区别",
    "差异",
)
MODEL_FIRST_INTENTS = {"case_reasoning", "comparison", "penalty_elements"}
MIXED_MODEL_INTENTS = {"article_locate", "procedure_extract", "contract_clause", "case_holding"}
LEGAL_QA_SYSTEM_PROMPT = """
你是法律 RAG 问答助手，只能依据用户给出的检索证据作答。
回答策略：
1. 先给结论，再给最小必要依据；每个事实性结论必须带 ##n$$ 引用。
2. 必须覆盖问题中的硬约束词、引号内短语、条号、案号、日期、版次和并列对象。
3. 强语义题按“结论 -> 关键依据 -> 关键词命中”组织；并列比较题按“对象A -> 对象B -> 差异点/共同点”组织。
4. 不得无证据扩写，不得用相似文档替代目标文档；证据不足时明确拒答并说明缺失项。

Few-shot:
Q: 请解释《民法典》第七条的含义，必须出现“诚实信用”。
A: 结论：《民法典》第七条确立的是民事主体从事民事活动应遵循诚实信用原则。##1$$
关键依据：条文直接要求民事主体从事民事活动应当遵循该原则。##1$$
关键词命中：第七条、诚实信用。

Q: 请比较甲法和乙条例对办理时限的差异。
A: 结论：甲法侧重设定基本时限，乙条例进一步细化办理节点。##1$$ ##2$$
对象A：甲法规定基本时限。##1$$
对象B：乙条例细化受理、审查或办结节点。##2$$
差异点：一个给出上位规则，一个给出执行层面的流程要求。
""".strip()


def create_client() -> OpenAI:
    return OpenAI(
        api_key=GENERATION_API_KEY,
        base_url=GENERATION_BASE_URL,
    )


def strip_code_fence(content: str) -> str:
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.search(pattern, content.strip(), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else content.strip()


def has_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in keywords)


def normalize_citation_markers(answer: str) -> str:
    if not answer:
        return answer

    normalized = re.sub(
        r"##((?:\[\d+\])+)\$\$",
        lambda match: " ".join(f"##{num}$$" for num in re.findall(r"\d+", match.group(1))),
        answer,
    )
    normalized = re.sub(r"##(\d+)##", lambda match: f"##{match.group(1)}$$", normalized)
    normalized = re.sub(r"##(?:reference|citation)\[(\d+)\]\$\$", lambda match: f"##{match.group(1)}$$", normalized)
    normalized = re.sub(r"##(?:reference|citation)[_ ]?(\d+)##", lambda match: f"##{match.group(1)}$$", normalized)
    normalized = re.sub(r"##(?:reference|citation)[_ ]?(\d+)\$\$", lambda match: f"##{match.group(1)}$$", normalized)
    normalized = re.sub(r"(?<![\w(])\[(\d+)\](?!\()", lambda match: f"##{match.group(1)}$$", normalized)
    return normalized


def build_citation_key_to_index(references: list[dict]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for reference in references:
        citation_key = str(reference.get("rule_citation_key", "")).strip()
        citation_index = int(reference.get("citation_index", 0) or 0)
        if citation_key and citation_index and citation_key not in mapping:
            mapping[citation_key] = citation_index
    return mapping


def remap_rule_citation_markers(answer: str, retrieved_content) -> str:
    if not answer:
        return answer
    references = normalize_references(retrieved_content)
    key_to_index = build_citation_key_to_index(references)

    def replace(match: re.Match[str]) -> str:
        citation_index = key_to_index.get(match.group(1))
        return f"##{citation_index}$$" if citation_index else ""

    remapped = RULE_CITATION_PATTERN.sub(replace, answer)
    return normalize_citation_markers(remapped)


def first_reference_page(reference: dict) -> int | None:
    page_num = reference.get("page_num")
    if isinstance(page_num, int) and page_num > 0:
        return page_num

    page_num_int = reference.get("page_num_int")
    if isinstance(page_num_int, list) and page_num_int:
        try:
            return int(page_num_int[0])
        except (TypeError, ValueError):
            return None

    positions = reference.get("positions") or []
    if positions and isinstance(positions[0], (list, tuple)) and positions[0]:
        try:
            return int(positions[0][0])
        except (TypeError, ValueError):
            return None
    return None


def build_reference_location_label(reference: dict) -> str:
    document_name = reference.get("document_name", "") or "Unknown document"
    page_num = first_reference_page(reference)
    if page_num:
        return f"{document_name} / page {page_num}"
    return document_name


def normalize_references(retrieved_content) -> list[dict]:
    normalized: list[dict] = []
    for index, reference in enumerate(retrieved_content or [], start=1):
        page_num = first_reference_page(reference)
        positions = reference.get("positions") or []
        rule_citation_key = citation_key_for_chunk(reference, fallback_index=index)
        normalized.append(
            {
                "id": str(reference.get("id", index)),
                "rank": int(reference.get("display_rank", index) or index),
                "display_rank": int(reference.get("display_rank", index) or index),
                "source_rank": int(reference.get("source_rank", reference.get("rank", index)) or index),
                "best_rank": int(reference.get("best_rank", reference.get("rank", index)) or index),
                "merge_hit_count": int(reference.get("merge_hit_count", 1) or 1),
                "citation_index": index,
                "citation_marker": f"##{index}$$",
                "rule_citation_key": rule_citation_key,
                "document_id": reference.get("document_id", ""),
                "chunk_id": reference.get("chunk_id", ""),
                "document_name": reference.get("document_name", ""),
                "content_with_weight": reference.get("content_with_weight", ""),
                "positions": positions,
                "page_num": page_num,
                "page_num_int": reference.get("page_num_int", []),
                "company": reference.get("company", ""),
                "report_period": reference.get("report_period", ""),
                "report_type": reference.get("report_type", ""),
                "source": reference.get("source", ""),
                "doc_type": reference.get("doc_type", ""),
                "authority": reference.get("authority", ""),
                "effective_or_revision_date": reference.get("effective_or_revision_date", ""),
                "similarity": float(reference.get("similarity", 0.0) or 0.0),
                "term_similarity": float(reference.get("term_similarity", 0.0) or 0.0),
                "vector_similarity": float(reference.get("vector_similarity", 0.0) or 0.0),
                "best_similarity": float(reference.get("best_similarity", reference.get("similarity", 0.0)) or 0.0),
                "fusion_score": float(reference.get("fusion_score", 0.0) or 0.0),
                "location_label": build_reference_location_label(reference),
            }
        )
    return normalized


def normalize_structured_facts(structured_facts: list[dict] | None, references: list[dict]) -> list[dict]:
    if not structured_facts:
        return []
    key_to_index = build_citation_key_to_index(references)
    normalized: list[dict] = []
    for fact in structured_facts:
        item = dict(fact)
        citation_key = str(item.get("citation_key", "")).strip()
        if citation_key and citation_key in key_to_index:
            item["citation_index"] = key_to_index[citation_key]
        normalized.append(item)
    return normalized


def extract_citation_indices(answer: str, reference_count: int) -> list[int]:
    indices: list[int] = []
    seen: set[int] = set()
    for match in CITATION_PATTERN.finditer(answer or ""):
        citation_index = int(match.group(1))
        if citation_index < 1 or citation_index > reference_count or citation_index in seen:
            continue
        seen.add(citation_index)
        indices.append(citation_index)
    return indices


def _has_conflicting_effective_dates(references: list[dict]) -> bool:
    dates = {
        str(item.get("effective_or_revision_date", "")).strip()
        for item in references
        if str(item.get("effective_or_revision_date", "")).strip()
    }
    return len(dates) >= 2


def _question_has_explicit_version_anchor(question: str) -> bool:
    q = question or ""
    if re.search(r"\b(?:19|20)\d{6}\b", q):
        return True
    return any(token in q for token in ("版本", "现行", "修订", "生效", "第", "期"))


def _question_version_anchor_covered(question: str, references: list[dict]) -> bool:
    if not references:
        return False
    q = (question or "").replace(" ", "")
    doc_text = _match_anchor_text("\n".join(str(item.get("document_name", "") or "") for item in references))

    compact_dates = re.findall(r"\b(?:19|20)\d{6}\b", q)
    if compact_dates:
        return any(d in doc_text for d in compact_dates)

    gazette_terms = re.findall(r"(20\d{2})年?第?([0-9一二三四五六七八九十]{1,2})期", q)
    if gazette_terms:
        zh_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
        def norm_issue(raw: str) -> str:
            return zh_map.get(raw, raw)
        fullwidth = str.maketrans("0123456789", "0123456789")
        for year, issue in gazette_terms:
            issue_n = norm_issue(issue)
            issue_fw = issue_n.translate(fullwidth)
            if year in doc_text and (f"_{issue_n}_" in doc_text or f"_{issue_fw}_" in doc_text or f"第{issue_n}期" in doc_text):
                return True
        return False

    return False


PAIR_DOC_HINTS = (
    "两个",
    "两份",
    "并列",
    "同时返回",
    "同时给出",
    "同时给",
    "新旧",
    "可比对",
    "法律+实施条例",
    "法律＋实施条例",
    "与实施条例",
    "和实施条例",
)
LOW_BAR_TITLE_HINTS = (
    "文件名",
    "表格文件名",
    "完整表名",
    "表名",
    "标题",
    "版本",
    "版次",
    "版本号",
    "版本数字",
    "哪份",
    "案由关键词",
    "纠纷性质",
    "责任类型",
    "罪名方向",
    "发布单位",
    "命中",
    "出现",
)
GENERIC_TITLE_ANCHORS = {"办事指南", "服务指南", "申请表", "申报表", "表单", "合同", "公报", "案例"}
TITLE_CANDIDATE_PATTERN = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9·（）()、]{2,80}?(?:法典|法律|法|实施条例|条例|规定|办法|公报|合同|指南|申请表|申报表|案例))"
)
GAZETTE_PAIR_PATTERN = re.compile(r"(20\d{2})年?第?([0-9一二三四五六七八九十]{1,2})期")
QUERY_NOISE_PREFIXES = (
    "请",
    "请并列",
    "请同时",
    "请回答",
    "请给",
    "请提供",
    "给出",
    "给",
    "给我",
    "我要",
    "答案",
    "并列",
    "同时",
    "如果有",
    "有没有",
    "和",
    "与",
    "返回",
    "定位",
    "检索",
    "回答",
    "提供",
)
QUERY_NOISE_TOKENS = {
    "请",
    "给",
    "我",
    "你",
    "帮",
    "返回",
    "定位",
    "检索",
    "提供",
    "回答",
    "答案",
    "直接",
    "精确",
    "具体",
    "对应",
    "文件",
    "文件名",
    "版本",
    "版次",
    "版本号",
    "版本数字",
    "关键词",
    "需",
    "需要",
    "包含",
    "出现",
    "命中",
    "不要",
    "泛泛解释",
    "全文",
    "完整版",
}
COMMON_QUERY_CHARS = set("请给我你帮的了和与并或在中对应哪份哪个什么是否有没有如果回答答案直接精确具体文件版本版次数字关键词需要需出现命中包含返回提供定位检索")
FULLWIDTH_DIGIT_TO_ASCII = str.maketrans("0123456789", "0123456789")


def _compact_anchor(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _match_anchor_text(value: str) -> str:
    return _compact_anchor(value).translate(FULLWIDTH_DIGIT_TO_ASCII)


def _query_requires_parallel_docs(question: str) -> bool:
    q = question or ""
    if any(token in q for token in PAIR_DOC_HINTS):
        return True
    if len(GAZETTE_PAIR_PATTERN.findall(q)) >= 2:
        return True
    if re.search(r"第[0-9一二三四五六七八九十]{1,2}期和第[0-9一二三四五六七八九十]{1,2}期", q):
        return True
    return False


def _strip_query_noise(term: str) -> str:
    cleaned = _compact_anchor(term).strip("，。；：:、（）() ")
    changed = True
    while changed:
        changed = False
        for prefix in QUERY_NOISE_PREFIXES:
            if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 1:
                cleaned = cleaned[len(prefix):]
                changed = True
    return cleaned.strip("，。；：:、（）() ")


def _raw_title_terms(question: str) -> list[str]:
    q = question or ""
    terms = [_strip_query_noise(item) for item in re.findall(r"《([^》]{2,80})》", q)]
    for candidate in TITLE_CANDIDATE_PATTERN.findall(q):
        cleaned = _strip_query_noise(candidate)
        if cleaned and cleaned not in QUERY_NOISE_TOKENS:
            terms.append(cleaned)

    ordered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if not term or term in seen or term in QUERY_NOISE_TOKENS:
            continue
        seen.add(term)
        ordered.append(term)
    return ordered


def _title_variants(term: str) -> list[str]:
    compact = _match_anchor_text(term)
    variants = [compact]
    replacements = (
        ("村委会", "村民委员会"),
        ("村民委员会", "村委会"),
        ("居委会", "城市居民委员会"),
        ("城市居民委员会", "居委会"),
        ("全国人大常委会", "全国人民代表大会常务委员会"),
        ("全国人民代表大会常务委员会", "全国人大常委会"),
        ("劳动争议法", "劳动合同法"),
        ("劳动争议", "劳动合同"),
    )
    for source, target in replacements:
        if source in compact:
            variants.append(compact.replace(source, target))
    if "市" in compact and len(compact) >= 6:
        variants.append(compact.replace("市", ""))
    return [item for item in dict.fromkeys(variants) if item]


def _specific_title_core(term: str) -> str:
    core = term.replace("中华人民共和国", "")
    for suffix in ("实施条例", "办事指南", "服务指南", "申请表", "申报表", "合同", "条例", "公报", "案例", "法"):
        if core.endswith(suffix) and len(core) > len(suffix):
            core = core[: -len(suffix)]
            break
    return core.strip()


def _reference_doc_text(references: list[dict]) -> str:
    return _match_anchor_text("\n".join(str(ref.get("document_name", "") or "") for ref in references))


def _anchor_in_doc_titles(anchor: str, references: list[dict]) -> bool:
    doc_text = _reference_doc_text(references)
    return any(variant in doc_text for variant in _title_variants(anchor))


def _title_anchor_covered(anchor: str, references: list[dict]) -> bool:
    variants = _title_variants(anchor)
    if not variants:
        return False
    doc_text = _reference_doc_text(references)
    if any(variant in doc_text for variant in variants):
        return True

    anchor_chars = set(variants[-1])
    if len(anchor_chars) < 4:
        return False
    for ref in references:
        doc = _match_anchor_text(str(ref.get("document_name", "") or ""))
        doc_no_city = doc.replace("市", "")
        for variant in variants:
            variant_no_city = variant.replace("市", "")
            specific_core = _specific_title_core(variant_no_city)
            if variant_no_city.startswith("中华人民共和国") and specific_core and specific_core not in doc_no_city:
                continue
            if variant_no_city and variant_no_city in doc_no_city:
                return True
            overlap = len(set(variant_no_city) & set(doc_no_city))
            if overlap >= max(4, int(len(set(variant_no_city)) * 0.72)):
                return True
    return False


def _best_title_overlap(question: str, references: list[dict]) -> int:
    chars = {
        ch
        for ch in _compact_anchor(question)
        if ("\u4e00" <= ch <= "\u9fff" or ch.isalnum()) and ch not in COMMON_QUERY_CHARS
    }
    if not chars:
        return 0
    best = 0
    for ref in references:
        doc_chars = set(_match_anchor_text(str(ref.get("document_name", "") or "")))
        best = max(best, len(chars & doc_chars))
    return best


def _has_low_bar_title_hit(question: str, references: list[dict]) -> bool:
    if not references:
        return False
    q = question or ""
    if not any(token in q for token in LOW_BAR_TITLE_HINTS):
        return False

    title_terms = _raw_title_terms(q)
    if title_terms and any(_anchor_in_doc_titles(term, references) for term in title_terms):
        return True
    if re.findall(r"\b(?:19|20)\d{6}\b", q) and _question_version_anchor_covered(q, references):
        return True
    if any(token in q for token in ("申请表", "申报表", "办事指南", "合同", "公报")) and _best_title_overlap(q, references) >= 4:
        return True
    return _best_title_overlap(q, references) >= 6


def _citation_binding_complete(answer: str, references: list[dict]) -> bool:
    citation_indices = extract_citation_indices(answer, len(references))
    if not citation_indices:
        return False
    for idx in citation_indices:
        ref = references[idx - 1]
        has_doc = bool(str(ref.get("document_name", "")).strip())
        has_locator = bool(ref.get("chunk_id") or ref.get("id") or ref.get("positions"))
        has_page = ref.get("page_num") is not None or bool(ref.get("page_num_int"))
        if not (has_doc and has_locator and has_page):
            return False
    return True


def _dedupe_ordered(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _match_anchor_text(value).strip("，。；：:、（）() ")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _year_phrase_variants(phrase: str) -> list[str]:
    compact = _match_anchor_text(phrase)
    variants = [compact]

    year_match = re.fullmatch(r"(20\d{2})年", compact)
    if year_match:
        y = year_match.group(1)
        variants.extend([y, f"{y}_1", f"{y}_2", f"{y}_3", f"{y}_4", f"{y}年"])

    issue_match = re.fullmatch(r"(20\d{2})[_-]([1-9]\d?)", compact)
    if issue_match:
        y, i = issue_match.groups()
        variants.extend([f"{y}_{i}", f"{y}年", y, f"第{i}期"])

    return _dedupe_ordered([item for item in variants if item])


def _phrase_hit(text: str, phrase: str) -> bool:
    hay = _match_anchor_text(text)
    needle = _match_anchor_text(phrase)
    if not needle:
        return True
    if needle in hay:
        return True

    for variant in _year_phrase_variants(phrase):
        if variant and variant in hay:
            return True
    return False


def _contains_any_variant(text: str, variants: list[str]) -> bool:
    return any(_phrase_hit(text, variant) for variant in variants if variant)


def _title_answer_variants(title: str) -> list[str]:
    variants = _title_variants(title)
    compact = _match_anchor_text(title)
    core = _specific_title_core(compact)
    if core and len(core) >= 2:
        variants.append(core)
        for suffix in ("法典", "法律", "法", "条例", "规定", "办法", "合同", "指南", "申请表", "申报表", "案例", "公报"):
            variants.append(f"{core}{suffix}")
    return _dedupe_ordered(variants)


def _extract_quoted_phrases(question: str) -> list[str]:
    q = question or ""
    quoted = re.findall(r"[“\"'《]([^”\"'》]{2,60})[”\"'》]", q)
    return _dedupe_ordered([item for item in quoted if item not in GENERIC_TITLE_ANCHORS])


def _extract_required_version_phrases(question: str) -> list[str]:
    q = question or ""
    phrases = []
    phrases.extend(re.findall(r"\b(?:19|20)\d{6}\b", q))
    phrases.extend(re.findall(r"(?:19|20)\d{2}版", q))
    phrases.extend(re.findall(r"(?:19|20)\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?", q))
    for year, issue in GAZETTE_PAIR_PATTERN.findall(q):
        zh_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9", "十": "10"}
        issue_no = zh_map.get(issue, issue)
        issue_full = issue_no.translate(str.maketrans("0123456789", "0123456789"))
        phrases.append(f"{year}_{issue_full}")
    return _dedupe_ordered(phrases)


def _extract_case_number_phrases(question: str) -> list[str]:
    return _dedupe_ordered(re.findall(r"指导性案例\s*第?\d+号|指导性案例\d+号", question or ""))


def _extract_answer_constraints(question: str) -> dict[str, list]:
    profile = classify_query_intent(question)
    all_phrases: list[str] = []
    title_groups: list[list[str]] = []

    all_phrases.extend(profile.article_terms)
    all_phrases.extend(_extract_case_number_phrases(question))
    all_phrases.extend(_extract_required_version_phrases(question))

    q = question or ""
    for quoted in _extract_quoted_phrases(q):
        if any(token in q for token in ("需", "必须", "命中", "包含", "出现", "是否出现", "答案要", "答案需", "同时出现")):
            all_phrases.append(quoted)

    for title in _raw_title_terms(q):
        if title in GENERIC_TITLE_ANCHORS:
            continue
        variants = _title_answer_variants(title)
        if variants:
            title_groups.append(variants)

    return {
        "required_all": _dedupe_ordered(all_phrases),
        "required_any_groups": title_groups,
    }


def _is_high_certainty_extraction(question: str) -> bool:
    q = question or ""
    if any(token in q for token in HIGH_CERTAINTY_HINTS):
        return True
    if any(token in q for token in ("只要", "直接给", "精确到", "精确返回")) and not any(token in q for token in ("解释", "理由", "差异", "区别")):
        return True
    return False


def _is_semantic_question(question: str) -> bool:
    q = question or ""
    semantic_view = q.replace("不要泛泛解释", "").replace("不要解释", "").replace("不需要解释", "")
    if any(token in semantic_view for token in SEMANTIC_GENERATION_HINTS):
        return True
    return classify_query_intent(semantic_view).intent in MODEL_FIRST_INTENTS


def classify_answer_route(question: str) -> dict:
    profile = classify_query_intent(question)
    q = question or ""
    high_certainty = _is_high_certainty_extraction(q)
    semantic = _is_semantic_question(q)

    if any(token in q for token in ("有没有", "是否有", "是否存在", "找不到", "没有的话")):
        return {
            "question_class": "A",
            "selected_mode": ROUTE_MODE_RULE,
            "fallback_mode": ROUTE_MODE_NONE,
            "route_reason": "abstain_guarded_existence_query",
            "intent": profile.intent,
        }

    # 高确定性抽取题强制 rule-first，保留 model fallback
    if (
        high_certainty
        or _query_requires_parallel_docs(q)
        or any(token in q for token in ("文件名", "版本号", "版次", "哪份", "案由关键词", "并列返回", "同时返回", "只输出文件名"))
        or (profile.intent == "article_locate" and not any(token in q for token in ("解释", "含义", "理由", "比较", "对比", "区别")))
    ):
        return {
            "question_class": "A",
            "selected_mode": ROUTE_MODE_RULE,
            "fallback_mode": ROUTE_MODE_MODEL,
            "route_reason": f"rule_first_high_certainty:{profile.intent}",
            "intent": profile.intent,
        }

    if semantic and not high_certainty:
        return {
            "question_class": "B",
            "selected_mode": ROUTE_MODE_MODEL,
            "fallback_mode": ROUTE_MODE_RULE,
            "route_reason": f"semantic_intent:{profile.intent}",
            "intent": profile.intent,
        }

    if profile.intent in MODEL_FIRST_INTENTS:
        return {
            "question_class": "C",
            "selected_mode": ROUTE_MODE_MODEL,
            "fallback_mode": ROUTE_MODE_RULE,
            "route_reason": f"model_first_intent:{profile.intent}",
            "intent": profile.intent,
        }

    return {
        "question_class": "A",
        "selected_mode": ROUTE_MODE_RULE,
        "fallback_mode": ROUTE_MODE_MODEL,
        "route_reason": f"high_certainty_or_rule_extract:{profile.intent}",
        "intent": profile.intent,
    }


def _is_abstain_answer(answer: str) -> bool:
    return has_any_keyword(answer or "", ABSTAIN_MARKERS)


def _best_reference_overlap(question: str, references: list[dict]) -> int:
    return _best_title_overlap(question, references)


def _cited_reference_overlap(question: str, answer: str, references: list[dict]) -> int:
    best = 0
    for index in extract_citation_indices(answer, len(references)):
        best = max(best, _best_title_overlap(question, [references[index - 1]]))
    return best


def validate_answer_accuracy(
    question: str,
    answer: str,
    references: list[dict],
    route_decision: dict,
) -> dict:
    failures: list[str] = []
    normalized_answer = normalize_citation_markers(answer or "")
    constraints = _extract_answer_constraints(question)
    citation_indices = extract_citation_indices(normalized_answer, len(references))
    question_requires_pair = _query_requires_parallel_docs(question)

    if references and not citation_indices:
        failures.append("missing_citation")

    if _is_abstain_answer(normalized_answer):
        failures.append("unexpected_abstain")

    if question_requires_pair:
        answer_lines = [line.strip() for line in (normalized_answer or "").splitlines() if line.strip()]
        cited_docs = {
            str(references[idx - 1].get("document_name", "") or "").strip()
            for idx in citation_indices
            if 0 < idx <= len(references)
        }
        if len(answer_lines) < 2 and len(citation_indices) < 2 and len(cited_docs) < 2:
            failures.append("pair_documents_not_complete")

    for phrase in constraints["required_all"]:
        if not _phrase_hit(normalized_answer, phrase):
            failures.append(f"missing_required_phrase:{phrase}")

    for variants in constraints["required_any_groups"]:
        if not _contains_any_variant(normalized_answer, variants):
            failures.append(f"missing_title_anchor:{variants[0]}")

    if citation_indices and references:
        cited_doc_text = _match_anchor_text(
            "\n".join(str(references[idx - 1].get("document_name", "") or "") for idx in citation_indices if 0 < idx <= len(references))
        )
        softened: list[str] = []
        for failure in failures:
            if failure.startswith("missing_title_anchor:"):
                anchor = failure.split(":", 1)[1]
                if anchor and _contains_any_variant(cited_doc_text, _title_variants(anchor)):
                    continue
            softened.append(failure)
        failures = softened

    cited_best = _cited_reference_overlap(question, normalized_answer, references)
    global_best = _best_reference_overlap(question, references)
    if global_best >= 7 and cited_best + 3 < global_best:
        failures.append("cited_document_not_best_question_anchor")

    if route_decision.get("question_class") == "B":
        body = re.sub(CITATION_PATTERN, "", normalized_answer)
        body = re.sub(r"\s+", "", body)
        if len(body) < 24 or not any(token in normalized_answer for token in ("结论", "依据", "关键", "差异", "对象")):
            failures.append("semantic_answer_too_shallow")

    cited_text = "\n".join(
        f"{ref.get('document_name', '')}\n{ref.get('content_with_weight', '')}"
        for ref in references
        if ref
    )
    for phrase in constraints["required_all"]:
        if _phrase_hit(normalized_answer, phrase) and not _phrase_hit(cited_text, phrase):
            failures.append(f"required_phrase_not_in_evidence:{phrase}")

    return {
        "passed": not failures,
        "failures": _dedupe_ordered(failures),
        "constraints": constraints,
    }


def build_answer_audit(
    answer: str,
    references: list[dict],
    *,
    mode: str,
    retrieval_trace: dict | None = None,
    rule_reason: str | None = None,
    structured_facts: list[dict] | None = None,
    confidence: float | None = None,
    refusal_reason: str | None = None,
    route_reason: str | None = None,
    selected_mode: str | None = None,
    fallback_mode: str | None = None,
    answer_guard: dict | None = None,
) -> dict:
    normalized_answer = normalize_citation_markers(answer or "")
    citation_indices = extract_citation_indices(normalized_answer, len(references))
    supporting_chunks = [references[index - 1] for index in citation_indices]
    return {
        "mode": mode,
        "selected_mode": selected_mode,
        "fallback_mode": fallback_mode,
        "route_reason": route_reason,
        "rule_reason": rule_reason,
        "refusal_reason": refusal_reason,
        "confidence": confidence,
        "answer_guard": answer_guard or {},
        "citation_indices": citation_indices,
        "citation_binding_complete": _citation_binding_complete(normalized_answer, references),
        "conflicting_effective_dates": _has_conflicting_effective_dates(references),
        "structured_facts": structured_facts or [],
        "supporting_chunks": supporting_chunks,
        "supporting_documents": list(
            dict.fromkeys(chunk.get("document_name", "") for chunk in supporting_chunks if chunk.get("document_name"))
        ),
        "retrieval_trace": retrieval_trace or {},
    }


def serialize_documents_payload(
    retrieved_content,
    *,
    answer: str | None = None,
    mode: str | None = None,
    retrieval_trace: dict | None = None,
    rule_reason: str | None = None,
    structured_facts: list[dict] | None = None,
    confidence: float | None = None,
    refusal_reason: str | None = None,
    route_reason: str | None = None,
    selected_mode: str | None = None,
    fallback_mode: str | None = None,
    answer_guard: dict | None = None,
) -> dict:
    references = normalize_references(retrieved_content)
    normalized_structured_facts = normalize_structured_facts(structured_facts, references)
    payload = {
        "references": references,
        "retrieval_trace": retrieval_trace or {},
    }
    if answer is not None and mode is not None:
        payload["answer_audit"] = build_answer_audit(
            answer,
            references,
            mode=mode,
            retrieval_trace=retrieval_trace,
            rule_reason=rule_reason,
            structured_facts=normalized_structured_facts,
            confidence=confidence,
            refusal_reason=refusal_reason,
            route_reason=route_reason,
            selected_mode=selected_mode,
            fallback_mode=fallback_mode,
            answer_guard=answer_guard,
        )
    return payload


def _clip_text_for_prompt(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def format_references(retrieved_content) -> str:
    references = []
    normalized_refs = normalize_references(retrieved_content)[:MAX_PROMPT_REFERENCES]
    remaining_budget = min(MAX_PROMPT_TOTAL_CHARS, 3000)

    for idx, ref in enumerate(normalized_refs, start=1):
        metadata = []
        for key, label in (
            ("document_name", "document"),
            ("authority", "authority"),
            ("effective_or_revision_date", "effective_or_revision_date"),
            ("doc_type", "doc_type"),
            ("source", "source"),
        ):
            value = ref.get(key)
            if value:
                metadata.append(f"{label}={value}")

        meta_line = f"[{idx}] {'; '.join(metadata)}" if metadata else f"[{idx}]"
        content_budget = min(MAX_REFERENCE_CHARS, max(180, remaining_budget // max(1, (MAX_PROMPT_REFERENCES - idx + 1))))
        content = _clip_text_for_prompt(ref.get("content_with_weight", ""), content_budget)
        block = f"{meta_line}\n{content}".strip()

        if len(block) > remaining_budget and references:
            break
        if len(block) > remaining_budget:
            block = _clip_text_for_prompt(block, remaining_budget)

        references.append(block)
        remaining_budget -= len(block)
        if remaining_budget <= 80:
            break

    return "\n\n".join(references) if references else "No references available."


def detect_generation_task_type(question: str) -> str:
    profile = classify_query_intent(question)
    q = question or ""
    if any(token in q for token in ("有没有", "是否有", "是否存在", "如果有")):
        return "abstain_guarded"
    if _query_requires_parallel_docs(q):
        return "pair_exact"
    if profile.article_terms:
        return "law_article_numeric"
    if any(token in q for token in ("版本", "版次", "版本号", "版本数字", "现行", "修订")):
        return "version_numeric"
    if any(token in q for token in ("申请表", "申报表", "办事指南", "表名", "标题")):
        return "procedure_title"
    if profile.case_terms or "指导性案例" in q:
        return "case_title"
    return profile.intent


def format_generation_task_contract(question: str) -> str:
    task_type = detect_generation_task_type(question)
    contracts = {
        "law_article_numeric": (
            "Task type: law_article_numeric. Output only: document title + article number + exact keyword/original phrase + citation. "
            "Do not replace the precise field with general explanation."
        ),
        "version_numeric": (
            "Task type: version_numeric. Output only the complete retrieved file name that contains the requested date/version + citation."
        ),
        "procedure_title": (
            "Task type: procedure_title. Output only the retrieved guide/table/application title keywords + citation."
        ),
        "case_title": (
            "Task type: case_title. Output the retrieved case number and dispute/case title keywords + citation."
        ),
        "case_holding": (
            "Task type: legal_case_holding. Start with 结论, then give裁判要旨/关键词命中 from cited evidence."
        ),
        "case_reasoning": (
            "Task type: legal_semantic. Start with 结论, then map裁判理由/争议焦点/关键词 to cited evidence. "
            "Do not answer with only a file name."
        ),
        "penalty_elements": (
            "Task type: legal_elements. Extract and synthesize the required legal elements from 1-3 cited snippets."
        ),
        "comparison": (
            "Task type: legal_comparison. Answer by 对象A/对象B/差异点, using citations for each object."
        ),
        "pair_exact": (
            "Task type: pair_exact. Output exactly two lines; each line must be one retrieved file name plus its citation. "
            "Do not add a third document or substitute an unrelated nearby document."
        ),
        "abstain_guarded": (
            "Task type: abstain_guarded. If the requested entity is not present in retrieved titles/anchors, refuse. "
            "Never answer with a similar law, case, guide, or contract."
        ),
    }
    return contracts.get(
        task_type,
        "Task type: grounded_qa. Answer only from candidate anchors and cited snippets.",
    )


def extract_candidate_answer_anchors(question: str, retrieved_content) -> str:
    refs = normalize_references(retrieved_content)[:MAX_PROMPT_REFERENCES]
    if not refs:
        return ""

    anchors: list[str] = []
    requested_articles = classify_query_intent(question).article_terms
    for ref in refs:
        marker = ref.get("citation_marker", "")
        doc_name = str(ref.get("document_name", "") or "").strip()
        content = str(ref.get("content_with_weight", "") or "")
        parts: list[str] = []
        if doc_name:
            parts.append(f"title={doc_name}")
        for date in re.findall(r"\b(?:19|20)\d{6}\b|(?:19|20)\d{2}版", doc_name):
            parts.append(f"version={date}")
        for case in re.findall(r"指导性案例\s*第?\d+号|指导性案例\d+号", doc_name):
            parts.append(f"case={case}")
        for article in requested_articles:
            if article and article in content:
                article_index = content.find(article)
                snippet = _clip_text_for_prompt(content[article_index: article_index + 120], 120)
                parts.append(f"article={snippet}")
        if parts:
            anchors.append(f"{marker} " + "; ".join(parts))

    return "\n".join(anchors[:MAX_PROMPT_REFERENCES])


def build_answer_requirements(question: str) -> str:
    requirements = [
        "1) Use the same language as the user.",
        "2) Use only provided references; do not add unsupported facts.",
        "3) Give the conclusion first, then brief bullets if needed.",
        "4) Cite factual claims with markers like ##1$$.",
        "5) Keep numbers/dates/units exactly as in references.",
        "6) If evidence is insufficient, say what is missing and avoid guessing.",
        "7) Copy file names, version dates, article numbers, case numbers, and guide/table titles exactly from retrieved anchors.",
    ]

    if has_any_keyword(question, NUMERIC_QUERY_HINTS):
        requirements.append("8) For legal anchors, quote exact article/case/version wording.")
    if has_any_keyword(question, ANALYSIS_QUERY_HINTS):
        requirements.append("9) For analysis/comparison, provide 2-4 cited points.")
    if _is_semantic_question(question):
        requirements.append("10) Strong semantic legal answer format: 结论 -> 关键依据 -> 关键词命中.")
        requirements.append("11) If comparing, use 对象A -> 对象B -> 差异点 and cite both sides.")
    constraints = _extract_answer_constraints(question)
    flat_constraints = constraints["required_all"] + [group[0] for group in constraints["required_any_groups"] if group]
    if flat_constraints:
        requirements.append(f"12) Must cover these question constraints in the answer: {', '.join(flat_constraints[:10])}.")

    return "\n".join(requirements)


def generate_recommended_questions(user_question: str, retrieved_content=None) -> list[str]:
    document_names = []
    if retrieved_content:
        document_names = list(
            {
                item.get("document_name", "")
                for item in retrieved_content
                if item.get("document_name")
            }
        )[:3]

    context_line = f"Related documents: {', '.join(document_names)}" if document_names else ""
    prompt = f"""
You are a document QA assistant. Generate 3 useful follow-up questions for the user.
Use the same language as the user question.

User question: {user_question}
{context_line}

Requirements:
1. Keep questions specific and natural.
2. If document context exists, stay close to the document topic.
3. Return strict JSON with the field name recommended_questions.

Output format:
{{
  "recommended_questions": [
    "Question 1",
    "Question 2",
    "Question 3"
  ]
}}
""".strip()

    try:
        completion = create_client().chat.completions.create(
            model=RECOMMENDATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=30,
        )
        response = completion.choices[0].message.content or ""
        payload = json.loads(strip_code_fence(response))
        questions = payload.get("recommended_questions", [])
        if not isinstance(questions, list):
            return []
        return [item.strip() for item in questions if isinstance(item, str) and item.strip()][:3]
    except Exception:
        logger.exception("generate_recommended_questions failed")
        return []


def generate_session_name(user_question: str) -> str:
    prompt = f"""
Generate a short session title for the question below.
Use the same language as the user question.

User question: {user_question}

Requirements:
1. Keep the title concise.
2. Return strict JSON with the field name session_name.

Output format:
{{
  "session_name": "Session title"
}}
""".strip()

    try:
        completion = create_client().chat.completions.create(
            model=SESSION_NAME_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            stream=False,
            timeout=30,
        )
        response = completion.choices[0].message.content or ""
        payload = json.loads(strip_code_fence(response))
        session_name = payload.get("session_name")
        if isinstance(session_name, str) and session_name.strip():
            return session_name.strip()
    except Exception:
        logger.exception("generate_session_name failed")

    return user_question[:24]


def persist_chat_turn(
    session_id: str,
    user_id: str,
    user_question: str,
    model_answer: str,
    retrieval_content,
    recommended_questions,
    think: str,
) -> None:
    db = next(get_db())
    try:
        existing = db.execute(
            text(
                """
                SELECT user_id
                FROM sessions
                WHERE session_id = :session_id
                """
            ),
            {"session_id": session_id},
        ).fetchone()
        if existing:
            if existing.user_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="Session does not belong to the current user.",
                )
            db.execute(
                text(
                    """
                    UPDATE sessions
                    SET updated_at = NOW()
                    WHERE session_id = :session_id
                    """
                ),
                {"session_id": session_id},
            )
        else:
            db.execute(
                text(
                    """
                    INSERT INTO sessions (session_id, user_id, session_name)
                    VALUES (:session_id, :user_id, :session_name)
                    """
                ),
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "session_name": generate_session_name(user_question),
                },
            )

        db.execute(
            text(
                """
                INSERT INTO messages (
                    session_id,
                    user_question,
                    model_answer,
                    documents,
                    recommended_questions,
                    think
                )
                VALUES (
                    :session_id,
                    :user_question,
                    :model_answer,
                    :documents,
                    :recommended_questions,
                    :think
                )
                """
            ),
            {
                "session_id": session_id,
                "user_question": user_question,
                "model_answer": model_answer,
                "documents": json.dumps(retrieval_content or [], ensure_ascii=False),
                "recommended_questions": json.dumps(recommended_questions or [], ensure_ascii=False),
                "think": think,
            },
        )
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist chat history: {exc}",
        ) from exc
    finally:
        db.close()


def build_prompt(
    question: str,
    retrieved_content,
    history_turns=None,
    standalone_query: str | None = None,
) -> str:
    formatted_references = format_references(retrieved_content)
    requirements = build_answer_requirements(question)
    task_contract = format_generation_task_contract(question)
    candidate_anchors = extract_candidate_answer_anchors(question, retrieved_content)
    profile = classify_query_intent(question)
    expected_doc_count = 2 if _query_requires_parallel_docs(question) else 1

    planner_constraints = (
        f"intent={profile.intent}; "
        f"article_terms={list(profile.article_terms)}; "
        f"case_terms={list(profile.case_terms)}; "
        f"date_terms={list(profile.date_terms)}; "
        f"expected_doc_count={expected_doc_count}; "
        f"needs_abstain_guard={profile.needs_abstain_guard}"
    )

    abstain_policy = (
        "Abstain if required anchors are missing, required parallel docs < 2, or request is outside corpus. "
        "Format: 结论/原因(含##n$$)/缺失信息。"
    )

    prompt_sections = [
        "Rules:",
        requirements,
        "Task Contract:",
        task_contract,
        f"Planner: {planner_constraints}",
        f"Abstain: {abstain_policy}",
    ]

    if candidate_anchors:
        prompt_sections.extend([
            "Candidate Answer Anchors:",
            candidate_anchors,
        ])

    history_block = _clip_text_for_prompt(format_history_for_prompt(history_turns), 900)
    if history_block:
        prompt_sections.extend(["History:", history_block])

    prompt_sections.extend([
        "References:",
        formatted_references,
        "Question:",
        question,
    ])
    prompt = "\n\n".join(prompt_sections).strip()

    if standalone_query and standalone_query.strip() and standalone_query.strip() != question.strip():
        prompt += f"\n\nRetrieval query used:\n{_clip_text_for_prompt(standalone_query.strip(), 180)}"

    prompt = _clip_text_for_prompt(prompt, 5600)
    return prompt


def _detect_case_type(question: str) -> str:
    q = question or ""
    if "指导性案例" in q:
        return "case_id_text"
    if _query_requires_parallel_docs(q) or any(token in q for token in ("对比", "比较")):
        return "pair"
    if any(token in q for token in ("版本", "版", "日期", "202", "201")):
        return "version"
    if any(token in q for token in ("第", "条", "款", "项")):
        return "article"
    return "generic"


def _render_structured_answer(question: str, references: list[dict], fallback_answer: str) -> str:
    case_type = _detect_case_type(question)
    q = question or ""

    # 仅基于问题锚点选择文档，避免“取前两条”导致答案命中文档错误。
    title_terms = re.findall(r"《([^》]{2,80})》", q)
    date_terms = re.findall(r"\b\d{8}\b", q)
    gazette_terms = re.findall(r"(20\d{2})年?第?([0-9一二三四五六七八九十]{1,2})期", q)

    fullwidth_digit_map = str.maketrans("0123456789", "0123456789")

    def _normalize_issue_no(raw: str) -> str:
        zh_map = {
            "一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
            "六": "6", "七": "7", "八": "8", "九": "9", "十": "10",
        }
        return zh_map.get(raw, raw)

    unique_docs: list[tuple[int, str]] = []
    seen_docs: set[str] = set()
    for idx, ref in enumerate(references, start=1):
        doc = str(ref.get("document_name", "") or "").strip()
        if not doc or doc in seen_docs:
            continue
        seen_docs.add(doc)
        unique_docs.append((idx, doc))

    def _doc_score(doc_name: str) -> int:
        score = 0
        compact_doc = doc_name.replace(" ", "")
        for title in title_terms:
            if title and title in compact_doc:
                score += 6
        for d in date_terms:
            if d and d in compact_doc:
                score += 4
        for year, issue in gazette_terms:
            issue_norm = _normalize_issue_no(issue)
            issue_full = issue_norm.translate(fullwidth_digit_map)
            if year in compact_doc:
                score += 2
            if f"_{issue_norm}_" in compact_doc or f"_{issue_full}_" in compact_doc:
                score += 8
        if "指导性案例" in q and "指导性案例" in compact_doc:
            score += 3
        if any(token in q for token in ("实施条例",)) and "实施条例" in compact_doc:
            score += 3
        return score

    ranked_docs = sorted(unique_docs, key=lambda item: (_doc_score(item[1]), -item[0]), reverse=True)

    if case_type == "pair" and ranked_docs:
        selected = ranked_docs[:2]
        if len(selected) >= 2:
            return "\n".join(f"{doc} ##{idx}$$" for idx, doc in selected)
    if case_type == "version" and ranked_docs:
        idx, doc = ranked_docs[0]
        return f"{doc} ##{idx}$$"
    if case_type == "case_id_text" and ranked_docs:
        idx, doc = ranked_docs[0]
        return f"{doc.replace('.pdf','')} ##{idx}$$"

    if any(token in q for token in ("第", "条", "款", "项")):
        return fallback_answer

    if title_terms:
        for idx, doc in ranked_docs:
            if any(title in doc for title in title_terms):
                return f"{doc} ##{idx}$$"

    return fallback_answer


def evaluate_answerability(question: str, references: list[dict]) -> tuple[bool, str, list[str]]:
    profile = classify_query_intent(question)
    if not references:
        return False, "no_references", ["未检索到任何参考证据"]

    ref_text = "\n".join(str(ref.get("content_with_weight", "") or "") for ref in references)
    ref_docs = [str(ref.get("document_name", "") or "") for ref in references]
    unique_docs = {doc for doc in ref_docs if doc}

    normalized_docs = _match_anchor_text("\n".join(ref_docs))
    normalized_text = _match_anchor_text(ref_text)

    def _contains_anchor(term: str, *, include_text: bool = True) -> bool:
        anchor = _match_anchor_text(term)
        if not anchor:
            return False
        return anchor in normalized_docs or (include_text and anchor in normalized_text)

    missing: list[str] = []
    for article in profile.article_terms:
        if article and not _contains_anchor(article):
            missing.append(f"缺少条款锚点: {article}")
    for case in profile.case_terms:
        if case and not _contains_anchor(case, include_text=False):
            missing.append(f"缺少案例锚点: {case}")
    for date in profile.date_terms:
        compact = date.replace("年", "").replace("月", "").replace("日", "")
        if not _contains_anchor(date, include_text=False) and compact and not _contains_anchor(compact, include_text=False):
            missing.append(f"缺少版本日期锚点: {date}")

    cleaned_title_terms: list[str] = []
    covered_title_terms: list[str] = []
    for title in list(profile.title_terms) + _raw_title_terms(question):
        title_norm = (title or "").replace(" ", "")
        if not title_norm:
            continue
        title_norm = _strip_query_noise(title_norm)
        if not title_norm or title_norm in cleaned_title_terms:
            continue
        cleaned_title_terms.append(title_norm)
        if _title_anchor_covered(title_norm, references):
            covered_title_terms.append(title_norm)
        else:
            missing.append(f"缺少目标文档标题锚点: {title}")

    if covered_title_terms and not _query_requires_parallel_docs(question):
        missing = [item for item in missing if not item.startswith("缺少目标文档标题锚点")]
    elif profile.case_terms and not any(item.startswith("缺少案例锚点") for item in missing):
        missing = [
            item
            for item in missing
            if not (
                item.startswith("缺少目标文档标题锚点")
                and item.split(":", 1)[-1].strip() in {"需含案例", "案号", "指导性案例"}
            )
        ]

    if profile.needs_exact_match and not cleaned_title_terms and not profile.case_terms and not profile.date_terms:
        best_overlap = _best_title_overlap(question, references)
        if best_overlap < 4 and _question_has_explicit_version_anchor(question):
            if not _question_version_anchor_covered(question, references):
                missing.append("缺少可验证的目标文档名称锚点")
        elif best_overlap < 4 and not _has_low_bar_title_hit(question, references):
            missing.append("缺少可验证的目标文档名称锚点")

    if _has_low_bar_title_hit(question, references):
        missing = [
            item
            for item in missing
            if not (
                item.startswith("缺少可验证的目标文档名称锚点")
                or (not cleaned_title_terms and item.startswith("缺少目标文档标题锚点"))
                or (
                    item.startswith("缺少目标文档标题锚点")
                    and item.split(":", 1)[-1].strip() in GENERIC_TITLE_ANCHORS
                )
            )
        ]

    if _query_requires_parallel_docs(question) and len(unique_docs) < 2:
        missing.append("并列检索证据不足两个文档")

    if profile.needs_abstain_guard and any(token in question for token in ("有没有", "是否有", "是否存在")) and missing:
        return False, "existence_not_proved", missing

    if missing:
        return False, "anchor_missing", missing
    return True, "", []


def build_abstain_answer(reason: str, missing: list[str], has_documents: bool) -> str:
    citation = "##1$$" if has_documents else ""
    missing_text = "；".join(missing) if missing else "缺少目标实体对应证据"
    return (
        "结论：无法回答，无法依据当前语料作答。\n"
        f"原因：缺少目标实体对应证据（{reason or 'evidence_insufficient'}）{citation}\n"
        f"缺失信息：{missing_text}\n"
        "建议：请补充准确文档名、版本日期或案例编号。"
    ).strip()


def _route_kwargs(route_decision: dict) -> dict:
    return {
        "route_reason": route_decision.get("route_reason"),
        "selected_mode": route_decision.get("selected_mode"),
        "fallback_mode": route_decision.get("fallback_mode"),
    }


def _build_answer_result(
    *,
    answer: str,
    mode: str,
    documents: list[dict],
    retrieval_trace: dict | None,
    route_decision: dict,
    answer_guard: dict | None,
    rule_reason: str | None = None,
    structured_facts: list[dict] | None = None,
    confidence: float | None = None,
    refusal_reason: str | None = None,
) -> dict:
    documents_payload = serialize_documents_payload(
        documents,
        answer=answer,
        mode=mode,
        retrieval_trace=retrieval_trace,
        rule_reason=rule_reason,
        structured_facts=structured_facts,
        confidence=confidence,
        refusal_reason=refusal_reason,
        answer_guard=answer_guard,
        **_route_kwargs(route_decision),
    )
    return {
        "answer": answer,
        "mode": mode,
        "documents": documents,
        "documents_payload": documents_payload,
        "answer_audit": documents_payload.get("answer_audit", {}),
        "route_reason": route_decision.get("route_reason"),
        "selected_mode": route_decision.get("selected_mode"),
        "fallback_mode": route_decision.get("fallback_mode"),
        "answer_guard": answer_guard or {},
    }


def _append_guard_retry_prompt(prompt: str, previous_answer: str | None, answer_guard: dict | None) -> str:
    if not previous_answer or not answer_guard:
        return prompt
    failures = answer_guard.get("failures") or []
    if not failures:
        return prompt

    failure_text = "\n".join(f"- {item}" for item in failures[:6])
    targeted_fix = ""
    if any(item.startswith("missing_required_phrase:") for item in failures):
        targeted_fix = "补全问题中的必含短语（允许日期/期次等价写法），不要改写为缺失。"
    elif any(item.startswith("missing_title_anchor:") for item in failures):
        targeted_fix = "优先使用已引用文档标题中的精确名称（含法名别名），不要引入新文档。"
    elif "pair_documents_not_complete" in failures:
        targeted_fix = "严格输出两行，每行一个文档名+引用，不要第三行。"

    return (
        f"{prompt}\n\n"
        "Answer Accuracy Guard failed. Rewrite once and fix listed failures only.\n"
        f"Failures:\n{failure_text}\n"
        f"Focused fix: {targeted_fix or '修复失败项并保持答案最短。'}\n\n"
        f"Previous answer:\n{previous_answer}\n\n"
        "Rewrite requirements: short answer, ##n$$ citations, no extra explanation."
    )


def _build_model_messages(prompt: str, route_decision: dict) -> list[dict]:
    route_line = (
        f"Routing: selected_mode={route_decision.get('selected_mode')}; "
        f"fallback_mode={route_decision.get('fallback_mode')}; "
        f"reason={route_decision.get('route_reason')}; "
        f"question_class={route_decision.get('question_class')}"
    )
    return [
        {
            "role": "system",
            "content": f"{LEGAL_QA_SYSTEM_PROMPT}\n\n{route_line}",
        },
        {"role": "user", "content": prompt},
    ]


def _create_chat_completion(messages: list[dict], *, stream: bool, timeout: int | None = None):
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            kwargs = {
                "model": CHAT_MODEL,
                "messages": messages,
                "stream": stream,
            }
            if timeout is not None:
                kwargs["timeout"] = timeout
            return create_client().chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 12))
    raise RuntimeError(f"Failed to start chat completion after retries: {last_error}")


def _chat_completion_timeout_seconds() -> int:
    raw = os.getenv("CHAT_COMPLETION_TIMEOUT_SECONDS", str(DEFAULT_CHAT_COMPLETION_TIMEOUT_SECONDS))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_CHAT_COMPLETION_TIMEOUT_SECONDS


def _generate_model_answer(
    question: str,
    documents: list[dict],
    *,
    history_turns=None,
    standalone_query: str | None = None,
    route_decision: dict,
    previous_answer: str | None = None,
    previous_guard: dict | None = None,
) -> str:
    prompt = build_prompt(
        question,
        documents,
        history_turns=history_turns,
        standalone_query=standalone_query,
    )
    prompt = _append_guard_retry_prompt(prompt, previous_answer, previous_guard)
    messages = _build_model_messages(prompt, route_decision)
    completion = _create_chat_completion(messages, stream=False, timeout=_chat_completion_timeout_seconds())
    normalized_answer = normalize_citation_markers(completion.choices[0].message.content or "")
    if route_decision.get("question_class") != "B":
        normalized_answer = _render_structured_answer(question, documents, normalized_answer)
    return normalized_answer


def _guard_answer(question: str, answer: str, documents: list[dict], route_decision: dict) -> dict:
    references = normalize_references(documents)
    guard = validate_answer_accuracy(question, answer, references, route_decision)
    if STRICT_CITATION_BINDING and not _citation_binding_complete(answer, references):
        failures = list(guard.get("failures") or [])
        failures.append("citation_binding_incomplete")
        guard = dict(guard)
        guard["passed"] = False
        guard["failures"] = _dedupe_ordered(failures)
    return guard


def _is_rule_safe_fallback_question(question: str) -> bool:
    q = question or ""
    return (
        _query_requires_parallel_docs(q)
        or any(token in q for token in ("文件名", "版本", "版次", "版本号", "条", "第", "并列", "同时返回", "哪份", "只输出文件名"))
    )


def _finalize_rule_answer(
    *,
    question: str,
    all_documents: list[dict],
    rule_result: dict,
    retrieval_trace: dict | None,
    route_decision: dict,
) -> dict | None:
    rule_documents = list(rule_result.get("documents") or all_documents)
    normalized_answer = remap_rule_citation_markers(rule_result["answer"], rule_documents)
    refusal_reason = rule_result.get("refusal_reason")
    rule_reason = rule_result.get("rule_reason")

    if STRICT_CITATION_BINDING and not _citation_binding_complete(normalized_answer, normalize_references(rule_documents)):
        citation = "##1$$" if rule_documents else ""
        normalized_answer = (
            "结论：无法回答，当前证据引用绑定不完整。\n"
            f"原因：citation_binding_incomplete {citation}\n"
            "缺失信息：需要可定位到文档与页码/chunk的引用"
        ).strip()
        rule_reason = "abstain_citation_binding"
        refusal_reason = "citation_binding_incomplete"

    answer_guard = _guard_answer(question, normalized_answer, rule_documents, route_decision)
    if not answer_guard.get("passed"):
        return None

    return _build_answer_result(
        answer=normalized_answer,
        mode=ROUTE_MODE_RULE,
        documents=rule_documents,
        retrieval_trace=retrieval_trace,
        route_decision=route_decision,
        answer_guard=answer_guard,
        rule_reason=rule_reason,
        structured_facts=rule_result.get("structured_facts"),
        confidence=rule_result.get("confidence"),
        refusal_reason=refusal_reason,
    )


def _finalize_model_answer(
    *,
    question: str,
    all_documents: list[dict],
    history_turns,
    standalone_query: str | None,
    retrieval_trace: dict | None,
    route_decision: dict,
    previous_answer: str | None = None,
    previous_guard: dict | None = None,
) -> dict:
    model_answer = _generate_model_answer(
        question,
        all_documents,
        history_turns=history_turns,
        standalone_query=standalone_query,
        route_decision=route_decision,
        previous_answer=previous_answer,
        previous_guard=previous_guard,
    )
    answer_guard = _guard_answer(question, model_answer, all_documents, route_decision)
    if not answer_guard.get("passed"):
        rewritten_answer = _generate_model_answer(
            question,
            all_documents,
            history_turns=history_turns,
            standalone_query=standalone_query,
            route_decision=route_decision,
            previous_answer=model_answer,
            previous_guard=answer_guard,
        )
        rewritten_guard = _guard_answer(question, rewritten_answer, all_documents, route_decision)
        if rewritten_guard.get("passed"):
            model_answer = rewritten_answer
            answer_guard = rewritten_guard
        else:
            model_answer = build_abstain_answer(
                "answer_guard_failed",
                list(rewritten_guard.get("failures") or []),
                bool(all_documents),
            )
            answer_guard = rewritten_guard

    refusal_reason = None if answer_guard.get("passed") else "answer_guard_failed"
    return _build_answer_result(
        answer=model_answer,
        mode=ROUTE_MODE_MODEL,
        documents=all_documents,
        retrieval_trace=retrieval_trace,
        route_decision=route_decision,
        answer_guard=answer_guard,
        refusal_reason=refusal_reason,
    )


def run_answer_pipeline(
    question: str,
    retrieved_content,
    history_turns=None,
    standalone_query: str | None = None,
    retrieval_trace: dict | None = None,
) -> dict:
    all_documents = list(retrieved_content or [])
    route_decision = classify_answer_route(question)

    answerable, abstain_reason, missing = evaluate_answerability(question, all_documents)
    if (
        CONFLICT_ABSTAIN_ENABLED
        and _question_has_explicit_version_anchor(question)
        and _has_conflicting_effective_dates(normalize_references(all_documents))
        and not _question_version_anchor_covered(question, normalize_references(all_documents))
    ):
        answerable = False
        abstain_reason = "effective_date_conflict"
        missing = list(missing) + ["证据存在多个生效/修订日期，无法确认唯一适用版本"]

    if not answerable:
        abstain_route = dict(route_decision)
        abstain_route.update(
            {
                "selected_mode": ROUTE_MODE_RULE,
                "fallback_mode": ROUTE_MODE_NONE,
                "route_reason": f"answerability_guard:{abstain_reason or 'evidence_insufficient'}",
            }
        )
        abstain_answer = build_abstain_answer(abstain_reason, missing, bool(all_documents))
        answer_guard = {
            "passed": True,
            "failures": [],
            "constraints": _extract_answer_constraints(question),
        }
        return _build_answer_result(
            answer=abstain_answer,
            mode=ROUTE_MODE_RULE,
            documents=all_documents,
            retrieval_trace=retrieval_trace,
            route_decision=abstain_route,
            answer_guard=answer_guard,
            rule_reason="abstain_guard",
            confidence=0.98,
            refusal_reason=abstain_reason or "evidence_insufficient",
        )

    rule_result = maybe_answer_by_rule(question, all_documents, history_turns=history_turns)

    if route_decision["selected_mode"] == ROUTE_MODE_RULE:
        if rule_result:
            rule_answer = _finalize_rule_answer(
                question=question,
                all_documents=all_documents,
                rule_result=rule_result,
                retrieval_trace=retrieval_trace,
                route_decision=route_decision,
            )
            if rule_answer:
                return rule_answer

        if route_decision.get("fallback_mode") == ROUTE_MODE_MODEL:
            previous_answer = None
            previous_guard = None
            if rule_result:
                rule_documents = list(rule_result.get("documents") or all_documents)
                previous_answer = remap_rule_citation_markers(rule_result["answer"], rule_documents)
                previous_guard = _guard_answer(question, previous_answer, rule_documents, route_decision)
            return _finalize_model_answer(
                question=question,
                all_documents=all_documents,
                history_turns=history_turns,
                standalone_query=standalone_query,
                retrieval_trace=retrieval_trace,
                route_decision=route_decision,
                previous_answer=previous_answer,
                previous_guard=previous_guard,
            )

    if route_decision["selected_mode"] == ROUTE_MODE_MODEL:
        try:
            model_result = _finalize_model_answer(
                question=question,
                all_documents=all_documents,
                history_turns=history_turns,
                standalone_query=standalone_query,
                retrieval_trace=retrieval_trace,
                route_decision=route_decision,
            )
            if (
                not model_result.get("answer_guard", {}).get("passed", False)
                and rule_result
                and _is_rule_safe_fallback_question(question)
            ):
                rule_answer = _finalize_rule_answer(
                    question=question,
                    all_documents=all_documents,
                    rule_result=rule_result,
                    retrieval_trace=retrieval_trace,
                    route_decision=route_decision,
                )
                if rule_answer:
                    return rule_answer
            return model_result
        except Exception:
            if route_decision.get("fallback_mode") == ROUTE_MODE_RULE and rule_result:
                rule_answer = _finalize_rule_answer(
                    question=question,
                    all_documents=all_documents,
                    rule_result=rule_result,
                    retrieval_trace=retrieval_trace,
                    route_decision=route_decision,
                )
                if rule_answer:
                    return rule_answer
            raise

    fallback_answer = build_abstain_answer("answer_route_failed", ["未能生成通过校验的答案"], bool(all_documents))
    fallback_guard = {
        "passed": False,
        "failures": ["answer_route_failed"],
        "constraints": _extract_answer_constraints(question),
    }
    return _build_answer_result(
        answer=fallback_answer,
        mode=ROUTE_MODE_RULE,
        documents=all_documents,
        retrieval_trace=retrieval_trace,
        route_decision=route_decision,
        answer_guard=fallback_guard,
        rule_reason="answer_route_failed",
        refusal_reason="answer_route_failed",
    )


def get_chat_completion(
    session_id: str,
    user_id: str,
    question: str,
    retrieved_content,
    history_turns=None,
    standalone_query: str | None = None,
    retrieval_trace: dict | None = None,
):
    try:
        result = run_answer_pipeline(
            question,
            retrieved_content,
            history_turns=history_turns,
            standalone_query=standalone_query,
            retrieval_trace=retrieval_trace,
        )
        active_documents = list(result.get("documents") or retrieved_content or [])
        documents_payload = result.get("documents_payload") or serialize_documents_payload(
            active_documents,
            answer=result.get("answer", ""),
            mode=result.get("mode", ROUTE_MODE_MODEL),
            retrieval_trace=retrieval_trace,
        )
        initial_payload = serialize_documents_payload(
            active_documents,
            retrieval_trace=retrieval_trace,
        )
        yield (
            "event: message\n"
            f"data: {json.dumps({'documents': initial_payload['references'], 'retrieval_trace': initial_payload['retrieval_trace']}, ensure_ascii=False)}\n\n"
        )

        model_answer = result.get("answer", "")
        payload = {
            "role": "assistant",
            "content": model_answer,
            "thinking": False,
        }
        yield f"event: message\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield (
            "event: message\n"
            f"data: {json.dumps({'answer_audit': documents_payload.get('answer_audit', {})}, ensure_ascii=False)}\n\n"
        )

        recommended_questions = generate_recommended_questions(question, active_documents)
        if recommended_questions:
            yield (
                "event: message\n"
                f"data: {json.dumps({'recommended_questions': recommended_questions}, ensure_ascii=False)}\n\n"
            )

        persist_chat_turn(
            session_id=session_id,
            user_id=user_id,
            user_question=question,
            model_answer=model_answer,
            retrieval_content=documents_payload,
            recommended_questions=recommended_questions,
            think="",
        )
        yield "event: end\ndata: [DONE]\n\n"
    except Exception as exc:
        logger.exception("get_chat_completion failed")
        error_message = {
            "role": "error",
            "content": str(exc),
        }
        yield f"event: error\ndata: {json.dumps(error_message, ensure_ascii=False)}\n\n"
