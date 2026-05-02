from __future__ import annotations

import hashlib
import re
from typing import Any

from service.core.retrieval_intent import classify_query_intent

ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千0-9]+(?:编|章|节|条|款|项)")
CASE_PATTERN = re.compile(r"指导性案例\s*第?\d+号|\(?\d{4}\)?[\u4e00-\u9fa5]{1,6}\d{2,8}号")
PROCEDURE_PATTERN = re.compile(r"(申请条件|申请材料|办理流程|审查时限|受理机关|办理时限)")
CONTRACT_PATTERN = re.compile(r"(合同目的|权利义务|违约责任|解除条件|争议解决|适用法律)")
DEADLINE_PATTERN = re.compile(r"(\d+\s*(?:个)?(?:工作日|自然日|日|天|个月|月|年)|审查时限|办理时限|受理期限|截止日期)")
AMOUNT_PATTERN = re.compile(r"([人民币￥¥]?\s*\d+[\d,]*(?:\.\d+)?\s*(?:元|万元|亿元|万|亿))")
DATE_COMPACT_PATTERN = re.compile(r"\b(?:19|20)\d{6}\b")
YEAR_VERSION_PATTERN = re.compile(r"(?:19|20)\d{2}版|(?:19|20)\d{2}年")
QUOTE_PATTERN = re.compile(r"[“\"']([^“”\"']{2,40})[”\"']")
GAZETTE_PAIR_PATTERN = re.compile(r"(20\d{2})年?第?([0-9一二三四五六七八九十]{1,2})期")
TITLE_DIRECT_HINTS = (
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
    "哪类",
    "案由关键词",
    "纠纷性质",
    "责任类型",
    "罪名方向",
    "发布单位",
    "命中",
    "出现",
)
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
QUERY_NOISE_TOKENS = {
    "请",
    "给",
    "我",
    "你",
    "帮",
    "出",
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
}
COMMON_QUERY_CHARS = set("请给我你帮的了和与并或在中对应哪份哪个什么是否有没有如果回答答案直接精确具体文件版本版次数字关键词需要需出现命中包含返回提供定位检索")
UNSAFE_RULE_INTENTS = {
    "case_holding",
    "case_reasoning",
    "comparison",
    "doc_precise",
    "version_effect",
}


def clean_rule_text(value: str) -> str:
    text = str(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def citation_key_for_chunk(chunk: dict[str, Any], fallback_index: int | None = None) -> str:
    existing = str(chunk.get("_rule_citation_key", "") or chunk.get("rule_citation_key", "")).strip()
    if existing:
        return existing

    seed = "||".join(
        (
            str(chunk.get("document_id", "")),
            str(chunk.get("chunk_id", "")),
            clean_rule_text(str(chunk.get("document_name", ""))),
            clean_rule_text(str(chunk.get("content_with_weight", "")))[:240],
        )
    ).strip()
    if seed:
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    if fallback_index is not None:
        return f"local{fallback_index}"
    return "local0"


def citation_marker_for_chunk(chunk: dict[str, Any], fallback_index: int | None = None) -> str:
    return f"##ref_{citation_key_for_chunk(chunk, fallback_index=fallback_index)}$$"


def reindex_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for document in documents:
        key = (
            str(document.get("document_id", "")),
            str(document.get("chunk_id", "")),
            clean_rule_text(str(document.get("content_with_weight", "")))[:160],
        )
        if key in seen:
            continue
        seen.add(key)
        normalized_doc = dict(document)
        normalized_doc["id"] = len(normalized) + 1
        normalized_doc["_rule_citation_key"] = citation_key_for_chunk(normalized_doc, fallback_index=len(normalized) + 1)
        normalized.append(normalized_doc)
    return normalized


def _extract_best_evidence(chunks: list[dict[str, Any]], patterns: list[re.Pattern[str] | str]) -> tuple[dict[str, Any], str] | None:
    for chunk in chunks:
        text = clean_rule_text(str(chunk.get("content_with_weight", "")))
        if not text:
            continue
        for pattern in patterns:
            if isinstance(pattern, str):
                if pattern in text:
                    return chunk, text
            else:
                if pattern.search(text):
                    return chunk, text
    return None


def _rule_payload(
    *,
    answer: str,
    documents: list[dict[str, Any]],
    rule_reason: str,
    structured_facts: list[dict[str, Any]] | None = None,
    confidence: float | None = None,
    refusal_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "documents": reindex_documents(list(documents)),
        "rule_reason": rule_reason,
        "structured_facts": structured_facts or [],
        "confidence": confidence,
        "refusal_reason": refusal_reason,
    }


def _build_abstain(chunks: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    citation = citation_marker_for_chunk(chunks[0], fallback_index=1) if chunks else ""
    return _rule_payload(
        answer=(
            "结论：无法回答，无法依据当前语料作答。\n"
            f"原因：缺少目标实体对应证据（{reason}）{citation}\n"
            "建议：请补充准确文档名、版本日期或案例编号。"
        ).strip(),
        documents=chunks,
        rule_reason=reason,
        confidence=0.93,
        refusal_reason=reason,
    )


def _extract_document_name_from_question(query: str) -> str:
    m = re.search(r"《([^》]{2,80})》", query or "")
    if m:
        return m.group(1).strip()
    return ""


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _strip_pdf_suffix(document_name: str) -> str:
    return re.sub(r"\.pdf$", "", document_name or "", flags=re.IGNORECASE)


def _citation_for_doc(chunk: dict[str, Any]) -> str:
    return citation_marker_for_chunk(chunk)


def _fullwidth_issue(raw: str) -> str:
    zh_map = {
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "十": "10",
    }
    issue = zh_map.get(raw, raw)
    return issue.translate(str.maketrans("0123456789", "0123456789"))


def _query_requires_parallel_docs(query: str) -> bool:
    q = query or ""
    if any(token in q for token in PAIR_DOC_HINTS):
        return True
    if len(GAZETTE_PAIR_PATTERN.findall(q)) >= 2:
        return True
    if re.search(r"第[0-9一二三四五六七八九十]{1,2}期和第[0-9一二三四五六七八九十]{1,2}期", q):
        return True
    return False


def _strip_query_noise(term: str) -> str:
    cleaned = _compact_text(term).strip("，。；：:、（）() ")
    for prefix in ("请", "并列", "同时", "回答", "给出", "给", "返回", "输出", "命中", "出现"):
        if cleaned.startswith(prefix) and len(cleaned) > len(prefix) + 1:
            cleaned = cleaned[len(prefix):]
    return cleaned.strip("，。；：:、（）() ")


def _extract_query_terms(query: str) -> list[str]:
    q = query or ""
    terms: list[str] = []
    terms.extend(re.findall(r"《([^》]{2,80})》", q))
    terms.extend(QUOTE_PATTERN.findall(q))
    terms.extend(DATE_COMPACT_PATTERN.findall(q))
    terms.extend(YEAR_VERSION_PATTERN.findall(q))
    terms.extend(CASE_PATTERN.findall(q))

    aliases = {
        "村委会": "村民委员会",
        "居委会": "城市居民委员会",
        "法律+实施条例": "实施条例",
        "法律＋实施条例": "实施条例",
    }
    for source, target in aliases.items():
        if source in q:
            terms.append(target)

    for raw in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9（）()、]{2,40}", q):
        cleaned = raw.strip("，。；：:、（）() ")
        if not cleaned or cleaned in QUERY_NOISE_TOKENS:
            continue
        if any(noise in cleaned for noise in ("请", "答案", "回答", "需", "需要")) and len(cleaned) > 8:
            for delimiter in ("请", "答案", "回答", "需", "需要"):
                if delimiter in cleaned:
                    cleaned = cleaned.split(delimiter, 1)[0]
                    break
            cleaned = cleaned.strip("，。；：:、（）() ")
            if not cleaned:
                continue
        if any(
            suffix in cleaned
            for suffix in (
                "法",
                "法典",
                "合同",
                "指南",
                "申请表",
                "申报表",
                "案例",
                "条例",
                "公报",
                "组织法",
                "著作权",
                "知识产权",
            )
        ):
            terms.append(cleaned)

    ordered: list[str] = []
    seen: set[str] = set()
    for term in terms:
        compact = _strip_query_noise(term)
        if not compact or compact in QUERY_NOISE_TOKENS or compact in seen:
            continue
        seen.add(compact)
        ordered.append(compact)
        if "村委会" in compact:
            ordered.append(compact.replace("村委会", "村民委员会"))
        if "居委会" in compact:
            ordered.append(compact.replace("居委会", "城市居民委员会"))
        if "劳动争议" in compact:
            ordered.append(compact.replace("劳动争议", "劳动合同"))
    deduped: list[str] = []
    seen2: set[str] = set()
    for item in ordered:
        if not item or item in seen2:
            continue
        seen2.add(item)
        deduped.append(item)
    return deduped


def _doc_matches_explicit_anchor(query: str, doc_name: str) -> bool:
    q = query or ""
    doc = _compact_text(doc_name)
    title_terms = [_compact_text(item) for item in re.findall(r"《([^》]{2,80})》", q)]
    if title_terms and not any(term and term in doc for term in title_terms):
        return False

    case_terms = CASE_PATTERN.findall(q)
    if case_terms and not any(_compact_text(term) in doc for term in case_terms):
        return False

    compact_dates = DATE_COMPACT_PATTERN.findall(q)
    if compact_dates and not any(date in doc for date in compact_dates):
        return False

    year_versions = YEAR_VERSION_PATTERN.findall(q)
    if year_versions and not any(_compact_text(version).replace("年", "") in doc for version in year_versions):
        return False

    return True


def _doc_score(query: str, doc_name: str) -> int:
    q = query or ""
    doc = _compact_text(doc_name)
    score = 0
    years_in_query = re.findall(r"(20\d{2})年", q)
    for y in years_in_query:
        if y in doc:
            score += 12
        if re.search(rf"{y}_[1-9]", doc):
            score += 12
    for term in _extract_query_terms(q):
        if term in doc:
            score += min(30, 5 + len(term))
    for date in DATE_COMPACT_PATTERN.findall(q):
        if date in doc:
            score += 50
    for version in YEAR_VERSION_PATTERN.findall(q):
        compact_version = _compact_text(version).replace("年", "")
        if compact_version and compact_version in doc:
            score += 35
    for case in CASE_PATTERN.findall(q):
        if _compact_text(case) in doc:
            score += 60
    for year, issue in GAZETTE_PAIR_PATTERN.findall(q):
        issue_full = _fullwidth_issue(issue)
        if year in doc and f"_{issue_full}" in doc:
            score += 80
    if "申请表" in q and "申请表" in doc:
        score += 35
    if "申报表" in q and "申报表" in doc:
        score += 35
    if "办事指南" in q and "办事指南" in doc:
        score += 30
    if "实施条例" in q and "实施条例" in doc:
        score += 25
    if "合同" in q and "合同" in doc:
        score += 20
    if "公报" in q and "公报" in doc:
        score += 20
    query_chars = {
        ch
        for ch in _compact_text(q)
        if ("\u4e00" <= ch <= "\u9fff" or ch.isalnum()) and ch not in COMMON_QUERY_CHARS
    }
    if query_chars:
        score += min(20, len(query_chars & set(doc)))
    return score


def _select_title_document(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    allow_top_fallback: bool,
) -> tuple[dict[str, Any], str] | None:
    explicit_anchor = bool(
        re.findall(r"《([^》]{2,80})》", query or "")
        or CASE_PATTERN.findall(query or "")
        or DATE_COMPACT_PATTERN.findall(query or "")
        or YEAR_VERSION_PATTERN.findall(query or "")
    )

    ranked: list[tuple[int, int, dict[str, Any], str]] = []
    for index, chunk in enumerate(candidates):
        doc_name = clean_rule_text(str(chunk.get("document_name", "")))
        if not doc_name:
            continue
        if explicit_anchor and not _doc_matches_explicit_anchor(query, doc_name):
            continue
        score = _doc_score(query, doc_name)
        ranked.append((score, -index, chunk, doc_name))

    if ranked:
        ranked.sort(reverse=True)
        score, _, chunk, doc_name = ranked[0]
        if score > 0 or explicit_anchor or allow_top_fallback:
            return chunk, doc_name

    if allow_top_fallback and candidates:
        doc_name = clean_rule_text(str(candidates[0].get("document_name", "")))
        if doc_name:
            return candidates[0], doc_name
    return None


def _title_direct_task(query: str, profile) -> bool:
    q = query or ""
    if profile.article_terms:
        return False
    if profile.intent in {"version_effect", "doc_precise", "case_holding"}:
        return True
    if profile.intent in {"contract_clause", "procedure_extract"} and any(token in q for token in TITLE_DIRECT_HINTS):
        return True
    return any(token in q for token in TITLE_DIRECT_HINTS)


def _answer_title_direct_task(
    query: str,
    candidates: list[dict[str, Any]],
    profile,
) -> dict[str, Any] | None:
    if not _title_direct_task(query, profile):
        return None

    allow_top_fallback = not (
        re.findall(r"《([^》]{2,80})》", query or "")
        or CASE_PATTERN.findall(query or "")
        or DATE_COMPACT_PATTERN.findall(query or "")
        or YEAR_VERSION_PATTERN.findall(query or "")
    )
    selected = _select_title_document(query, candidates, allow_top_fallback=allow_top_fallback)
    if not selected:
        return None
    chunk, doc_name = selected
    answer_name = doc_name if "文件名" in (query or "") or "版本" in (query or "") or "版次" in (query or "") else _strip_pdf_suffix(doc_name)
    if "办事指南" in (query or "") and "办事指南" not in answer_name and "指南" in answer_name:
        answer_name = f"{answer_name}（办事指南）"
    answer = f"{answer_name} {_citation_for_doc(chunk)}"
    facts = [{"kind": "document", "document_name": doc_name, "citation_key": citation_key_for_chunk(chunk)}]
    return _rule_payload(answer=answer, documents=candidates, rule_reason="title_anchor_rule", structured_facts=facts, confidence=0.94)


def _answer_pair_title_task(query: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not _query_requires_parallel_docs(query):
        return None

    selected: list[tuple[dict[str, Any], str]] = []
    seen: set[str] = set()
    q = query or ""

    gazette_terms = GAZETTE_PAIR_PATTERN.findall(q)
    if gazette_terms:
        for year, issue in gazette_terms:
            issue_full = _fullwidth_issue(issue)
            match = None
            for chunk in candidates:
                doc_name = clean_rule_text(str(chunk.get("document_name", "")))
                doc_compact = _compact_text(doc_name)
                if doc_name in seen:
                    continue
                if year in doc_compact and f"_{issue_full}" in doc_compact:
                    match = (chunk, doc_name)
                    break
            if match:
                selected.append(match)
                seen.add(match[1])
        if len(selected) >= 2:
            return _build_pair_payload(selected[:2], candidates)

    if "法律+实施条例" in q or "法律＋实施条例" in q or "与实施条例" in q or "和实施条例" in q:
        base = None
        regulation = None
        title_terms = _extract_query_terms(q)
        for chunk in candidates:
            doc_name = clean_rule_text(str(chunk.get("document_name", "")))
            if not doc_name:
                continue
            score = _doc_score(q, doc_name)
            if score <= 0:
                continue
            if title_terms and not any(term in _compact_text(doc_name) for term in title_terms):
                continue
            if "实施条例" in doc_name and regulation is None:
                regulation = (chunk, doc_name)
            elif "法" in doc_name and "实施条例" not in doc_name and base is None:
                base = (chunk, doc_name)
        if base and regulation:
            return _build_pair_payload([base, regulation], candidates)
        return _build_abstain(candidates, "parallel_documents_missing")

    ranked: list[tuple[int, int, dict[str, Any], str]] = []
    for index, chunk in enumerate(candidates):
        doc_name = clean_rule_text(str(chunk.get("document_name", "")))
        if not doc_name:
            continue
        score = _doc_score(q, doc_name)
        if score > 0:
            ranked.append((score, -index, chunk, doc_name))
    ranked.sort(reverse=True)
    for score, _, chunk, doc_name in ranked:
        if doc_name in seen:
            continue
        selected.append((chunk, doc_name))
        seen.add(doc_name)
        if len(selected) >= 2:
            break

    if len(selected) >= 2:
        return _build_pair_payload(selected[:2], candidates)
    return _build_abstain(candidates, "parallel_documents_missing")


def _build_pair_payload(
    selected: list[tuple[dict[str, Any], str]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    top_two = selected[:2]
    answer = "\n".join(f"{doc_name} {_citation_for_doc(chunk)}" for chunk, doc_name in top_two)
    facts = [
        {
            "kind": "document_pair",
            "documents": [doc_name for _, doc_name in top_two],
            "citation_key": citation_key_for_chunk(top_two[0][0]) if top_two else "",
        }
    ]
    return _rule_payload(answer=answer, documents=candidates, rule_reason="pair_title_anchor_rule", structured_facts=facts, confidence=0.93)


def _extract_article_text(text: str, article_anchor: str) -> str:
    cleaned = clean_rule_text(text)
    if not article_anchor:
        return ""
    start = cleaned.find(article_anchor)
    if start < 0:
        return ""
    next_article = ARTICLE_PATTERN.search(cleaned, start + len(article_anchor))
    end = next_article.start() if next_article else min(len(cleaned), start + 180)
    snippet = cleaned[start:end].strip(" ：:。；;，,")
    return snippet[:180]


def _article_keyword(query: str, article_text: str) -> str:
    for quoted in QUOTE_PATTERN.findall(query or ""):
        if quoted and quoted in article_text:
            return quoted
    keyword_aliases = (
        ("诚实信用", ("诚实信用", "诚信原则", "诚信")),
        ("可以适用习惯", ("可以适用习惯",)),
    )
    for canonical, aliases in keyword_aliases:
        if any(alias in article_text for alias in aliases):
            return canonical
    body = ARTICLE_PATTERN.sub("", article_text, count=1).strip(" ：:。；;，,")
    return body[:36]


def maybe_answer_by_rule(
    query: str,
    chunks: list[dict[str, Any]],
    history_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    profile = classify_query_intent(query)
    candidates = reindex_documents(list(chunks))
    if not candidates:
        return None

    pair_result = _answer_pair_title_task(query, candidates)
    if pair_result:
        return pair_result

    if profile.intent == "article_locate":
        target_article = profile.article_terms[0] if profile.article_terms else ""
        selected = _select_title_document(query, candidates, allow_top_fallback=True)
        search_space = [selected[0]] if selected else candidates
        if selected:
            search_space.extend(chunk for chunk in candidates if chunk is not selected[0])

        for chunk in search_space:
            text = clean_rule_text(str(chunk.get("content_with_weight", "")))
            article_text = _extract_article_text(text, target_article) if target_article else ""
            if not article_text:
                continue
            keyword = _article_keyword(query, article_text)
            doc_name = clean_rule_text(str(chunk.get("document_name", "")))
            doc_label = _strip_pdf_suffix(doc_name)
            answer = f"{doc_label} {target_article} {keyword} {_citation_for_doc(chunk)}"
            facts = [
                {
                    "kind": "article",
                    "anchor": target_article,
                    "keyword": keyword,
                    "citation_key": citation_key_for_chunk(chunk),
                }
            ]
            return _rule_payload(answer=answer, documents=candidates, rule_reason="article_anchor_extract_rule", structured_facts=facts, confidence=0.94)

    title_result = _answer_title_direct_task(query, candidates, profile)
    if title_result:
        return title_result

    if profile.intent in UNSAFE_RULE_INTENTS:
        return None

    if profile.needs_abstain_guard and len(candidates) < 1:
        return _build_abstain(candidates, "insufficient_evidence")

    if profile.intent == "article_locate":
        evidence = _extract_best_evidence(candidates, [ARTICLE_PATTERN])
        if not evidence:
            return _build_abstain(candidates, "article_not_found")
        chunk, text = evidence
        article = ARTICLE_PATTERN.search(text)
        anchor = article.group(0) if article else "相关条文"
        answer = f"结论：{anchor}。依据：{citation_marker_for_chunk(chunk)}"
        facts = [{"kind": "article", "anchor": anchor, "citation_key": citation_key_for_chunk(chunk)}]
        return _rule_payload(answer=answer, documents=candidates, rule_reason="article_rule", structured_facts=facts, confidence=0.9)

    if profile.intent == "procedure_extract":
        evidence = _extract_best_evidence(candidates, [PROCEDURE_PATTERN])
        if not evidence:
            return _build_abstain(candidates, "procedure_not_found")
        chunk, text = evidence
        anchors = sorted(set(PROCEDURE_PATTERN.findall(text)))
        anchor_text = "、".join(anchors[:6]) if anchors else "申请要素"
        answer = f"结论：命中{anchor_text}。依据：{citation_marker_for_chunk(chunk)}"
        facts = [{"kind": "procedure", "anchors": anchors, "citation_key": citation_key_for_chunk(chunk)}]
        return _rule_payload(answer=answer, documents=candidates, rule_reason="procedure_rule", structured_facts=facts, confidence=0.89)

    if profile.intent == "contract_clause":
        evidence = _extract_best_evidence(candidates, [CONTRACT_PATTERN])
        if not evidence:
            return _build_abstain(candidates, "contract_clause_not_found")
        chunk, text = evidence
        anchors = sorted(set(CONTRACT_PATTERN.findall(text)))
        answer = f"结论：命中{('、'.join(anchors[:5]) or '关键条款')}。依据：{citation_marker_for_chunk(chunk)}"
        facts = [{"kind": "contract", "anchors": anchors, "citation_key": citation_key_for_chunk(chunk)}]
        return _rule_payload(answer=answer, documents=candidates, rule_reason="contract_rule", structured_facts=facts, confidence=0.89)

    if profile.is_deadline_query:
        evidence = _extract_best_evidence(candidates, [DEADLINE_PATTERN, PROCEDURE_PATTERN])
        if not evidence:
            return _build_abstain(candidates, "deadline_not_found")
        chunk, text = evidence
        deadlines = sorted(set(DEADLINE_PATTERN.findall(text)))
        deadline_text = "、".join(deadlines[:5]) if deadlines else "未抽取到明确时限描述"
        answer = f"结论：命中期限信息{deadline_text}。依据：{citation_marker_for_chunk(chunk)}"
        facts = [{"kind": "deadline", "values": deadlines, "citation_key": citation_key_for_chunk(chunk)}]
        return _rule_payload(answer=answer, documents=candidates, rule_reason="deadline_rule", structured_facts=facts, confidence=0.9)

    if profile.is_amount_query:
        evidence = _extract_best_evidence(candidates, [AMOUNT_PATTERN])
        if not evidence:
            return _build_abstain(candidates, "amount_not_found")
        chunk, text = evidence
        amounts = sorted(set(AMOUNT_PATTERN.findall(text)))
        amount_text = "、".join(amounts[:5]) if amounts else "未抽取到明确金额"
        answer = f"结论：命中金额信息{amount_text}。依据：{citation_marker_for_chunk(chunk)}"
        facts = [{"kind": "amount", "values": amounts, "citation_key": citation_key_for_chunk(chunk)}]
        return _rule_payload(answer=answer, documents=candidates, rule_reason="amount_rule", structured_facts=facts, confidence=0.9)

    if profile.intent in {"version_effect", "doc_precise"}:
        target_title = _extract_document_name_from_question(query)
        selected = None
        if target_title:
            for chunk in candidates:
                doc_name = clean_rule_text(str(chunk.get("document_name", "")))
                if target_title in doc_name:
                    selected = chunk
                    break
        if selected is None:
            selected = candidates[0]

        doc_name = clean_rule_text(str(selected.get("document_name", "")))
        if not doc_name:
            return _build_abstain(candidates, "doc_name_missing")

        answer = f"{doc_name} {citation_marker_for_chunk(selected)}"
        facts = [{"kind": "document", "document_name": doc_name, "citation_key": citation_key_for_chunk(selected)}]
        return _rule_payload(answer=answer, documents=candidates, rule_reason="doc_name_rule", structured_facts=facts, confidence=0.92)

    if profile.is_comparison:
        seen_docs: list[tuple[str, dict[str, Any]]] = []
        seen_names: set[str] = set()
        for chunk in candidates:
            doc_name = clean_rule_text(str(chunk.get("document_name", "")))
            if not doc_name or doc_name in seen_names:
                continue
            seen_names.add(doc_name)
            seen_docs.append((doc_name, chunk))
            if len(seen_docs) >= 2:
                break
        if len(seen_docs) >= 2:
            parts = [f"{name} {citation_marker_for_chunk(chunk)}" for name, chunk in seen_docs]
            facts = [{"kind": "document_pair", "documents": [name for name, _ in seen_docs], "citation_key": citation_key_for_chunk(seen_docs[0][1])}]
            return _rule_payload(answer="\n".join(parts), documents=candidates, rule_reason="doc_pair_rule", structured_facts=facts, confidence=0.9)

    return None
