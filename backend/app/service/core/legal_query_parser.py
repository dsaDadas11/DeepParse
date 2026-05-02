from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


FULLWIDTH_DIGITS = str.maketrans("0123456789", "0123456789")
MATCH_TRANSLATION = str.maketrans(
    "0123456789（）《》【】：，。；、",
    "0123456789()<>[]:,.;,",
)

CN_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
}

DOC_TYPE_HINTS = ("法", "条例", "实施条例", "公报", "指导性案例", "合同", "合同模板", "办事指南", "申请表", "申报表")
QUOTED_TITLE_PATTERN = re.compile(r"《([^》]{2,120})》")
DOC_TITLE_CANDIDATE_PATTERN = re.compile(
    r"([\u4e00-\u9fa5A-Za-z0-9·（）()、]{2,90}?"
    r"(?:法典|法律|法|实施条例|条例|规定|办法|公报|合同|指南|申请表|申报表|案例|标准))"
)
DATE8_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{6})(?!\d)")
DATE_TEXT_PATTERN = re.compile(r"((?:19|20)\d{2})年(?:(\d{1,2})月(?:(\d{1,2})日)?)?")
YEAR_PATTERN = re.compile(r"((?:19|20)\d{2})\s*年?")
ISSUE_PATTERN = re.compile(r"第\s*([0-90-9一二两三四五六七八九十百千]+)\s*期")
ARTICLE_PATTERN = re.compile(r"(?:第)?\s*([0-90-9一二两三四五六七八九十百千]+)\s*(条|款|项|章|节|编)")
CASE_NUMBER_PATTERN = re.compile(r"指导性案例\s*第?\s*(\d+)\s*号")
NEGATIVE_TITLE_PATTERN = re.compile(r"不要《([^》]{2,120})》(?:正文|本体|文件)?")
NEGATIVE_ISSUE_PATTERN = re.compile(r"不要\s*第\s*([0-90-9一二两三四五六七八九十百千]+)\s*期")
NEGATIVE_DATE_PATTERN = re.compile(r"不要[^，。；,.]{0,12}((?:19|20)\d{6})")


def normalize_digits(text: str) -> str:
    return (text or "").translate(FULLWIDTH_DIGITS)


def normalize_match_text(text: str) -> str:
    normalized = (text or "").translate(MATCH_TRANSLATION)
    normalized = normalized.lower()
    normalized = normalized.replace("中华人民共和国", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.strip()
    return normalized


def chinese_number_to_int(raw: str) -> int | None:
    value = normalize_digits(raw or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)

    total = 0
    section = 0
    digit = 0
    for char in value:
        num = CN_NUM_MAP.get(char)
        if num is None:
            return None
        if num >= 10:
            section += (digit or 1) * num
            digit = 0
        else:
            digit = digit * 10 + num
    total += section + digit
    return total if total > 0 else None


def _unique(values: list[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = (value or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def _normalize_issue(raw: str) -> str:
    number = chinese_number_to_int(raw)
    return str(number) if number is not None else normalize_digits(raw)


def _normalize_title(title: str) -> str:
    title = re.sub(r"^(?:请|给|我|你|帮|返回|定位|检索|提供|回答|并列|同时|不要|只要|答案)+", "", title or "")
    return title.strip("，。；：:、 ")


def _extract_dates(query: str) -> tuple[str, ...]:
    normalized = normalize_digits(query)
    dates = list(DATE8_PATTERN.findall(normalized))
    for year, month, day in DATE_TEXT_PATTERN.findall(normalized):
        if month and day:
            dates.append(f"{int(year):04d}{int(month):02d}{int(day):02d}")
    return _unique(dates)


def _extract_years(query: str) -> tuple[str, ...]:
    normalized = normalize_digits(query)
    years = [match for match in YEAR_PATTERN.findall(normalized)]
    return _unique([year for year in years if len(year) == 4])


def _extract_issues(query: str) -> tuple[str, ...]:
    return _unique([_normalize_issue(raw) for raw in ISSUE_PATTERN.findall(query)])


def _extract_negative_terms(query: str) -> tuple[str, ...]:
    normalized = normalize_digits(query)
    terms: list[str] = []
    for raw in re.findall(r"不要([^，。；,.]{1,28})", normalized):
        raw = raw.strip()
        if raw:
            terms.append(raw)
    if "不要基础版" in normalized:
        terms.append("基础版")
    if "不要带日期后缀" in normalized or "不要带日期" in normalized:
        terms.extend(["带日期后缀", "日期后缀"])
    return _unique(terms)


def _extract_titles(query: str) -> tuple[str, ...]:
    negative_titles = {_normalize_title(item) for item in NEGATIVE_TITLE_PATTERN.findall(query)}
    titles = [
        _normalize_title(item)
        for item in QUOTED_TITLE_PATTERN.findall(query)
        if _normalize_title(item) not in negative_titles
    ]
    if not titles and "公报" in query:
        if "全国人大常委会" in query or "全国人民代表大会常务委员会" in query:
            return ("全国人民代表大会常务委员会公报",)
        return ("公报",)
    if not titles:
        for candidate in DOC_TITLE_CANDIDATE_PATTERN.findall(query):
            title = _normalize_title(candidate)
            if title and title not in negative_titles and not title.startswith(("请", "给", "我要", "我只要", "我担心")):
                titles.append(title)
    return _unique([title for title in titles if title])


def _extract_subject_terms(query: str, titles: tuple[str, ...]) -> tuple[str, ...]:
    terms: list[str] = []
    normalized = query or ""
    if "自然保护区" in normalized and not any("自然保护区" in title for title in titles):
        terms.append("自然保护区")
    if "全国人大常委会公报" in normalized:
        terms.append("全国人民代表大会常务委员会公报")
    if "全国人民代表大会常务委员会公报" in normalized:
        terms.append("全国人民代表大会常务委员会公报")
    return _unique(terms)


def _contains_parallel_intent(query: str) -> bool:
    if any(token in query for token in ("同时", "并列", "对比", "比较")):
        return True
    if re.search(r"》\s*(?:与|和|及|以及)\s*《", query):
        return True
    if re.search(r"第\s*[0-90-9一二两三四五六七八九十百千]+\s*期\s*(?:与|和|及|以及)\s*第", query):
        return True
    if re.search(r"基础版\s*(?:与|和|及|以及)\s*(?:现行版|最新版|(?:19|20)\d{6})", normalize_digits(query)):
        return True
    return False


def _infer_route(
    query: str,
    titles: tuple[str, ...],
    years: tuple[str, ...],
    issues: tuple[str, ...],
    dates: tuple[str, ...],
    negative_terms: tuple[str, ...],
) -> str:
    if "公报" in query and (years or issues):
        if negative_terms or "同名" in query or "去歧义" in query:
            return "same_name_disambiguation"
        return "gazette_issue_lookup"
    if "基础版" in query and dates and _contains_parallel_intent(query):
        return "version_diff_lookup"
    if _contains_parallel_intent(query) and (len(titles) >= 2 or dates or "基础版" in query):
        return "version_conflict_lookup"
    if negative_terms or "同名" in query or "只要" in query or "不要" in query:
        return "same_name_disambiguation"
    if dates:
        return "version_lookup"
    return "general"


@dataclass(frozen=True)
class LegalQuerySlots:
    raw_query: str
    normalized_query: str
    route: str
    doc_type_terms: tuple[str, ...] = ()
    title_terms: tuple[str, ...] = ()
    subject_terms: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    issue_numbers: tuple[str, ...] = ()
    version_dates: tuple[str, ...] = ()
    article_terms: tuple[str, ...] = ()
    case_numbers: tuple[str, ...] = ()
    negative_terms: tuple[str, ...] = ()
    negative_title_terms: tuple[str, ...] = ()
    negative_issue_numbers: tuple[str, ...] = ()
    negative_version_dates: tuple[str, ...] = ()
    wants_base_version: bool = False
    wants_current_version: bool = False
    excludes_base_version: bool = False
    excludes_dated_version: bool = False
    parallel: bool = False
    expected_doc_count: int = 1
    target_doc_keys: tuple[str, ...] = field(default_factory=tuple)

    @property
    def strong_constraint(self) -> bool:
        return self.route in {
            "gazette_issue_lookup",
            "law_article_lookup",
            "same_name_disambiguation",
            "version_conflict_lookup",
            "version_diff_lookup",
            "version_lookup",
        }

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "title_terms": list(self.title_terms),
            "subject_terms": list(self.subject_terms),
            "years": list(self.years),
            "issue_numbers": list(self.issue_numbers),
            "version_dates": list(self.version_dates),
            "negative_terms": list(self.negative_terms),
            "negative_title_terms": list(self.negative_title_terms),
            "negative_issue_numbers": list(self.negative_issue_numbers),
            "wants_base_version": self.wants_base_version,
            "wants_current_version": self.wants_current_version,
            "excludes_base_version": self.excludes_base_version,
            "excludes_dated_version": self.excludes_dated_version,
            "parallel": self.parallel,
            "expected_doc_count": self.expected_doc_count,
            "target_doc_keys": list(self.target_doc_keys),
        }


def parse_legal_query(query: str) -> LegalQuerySlots:
    raw = (query or "").strip()
    normalized = normalize_digits(raw)
    titles = _extract_titles(raw)
    subject_terms = _extract_subject_terms(raw, titles)
    years = _extract_years(raw)
    negative_issues = _unique([_normalize_issue(raw_issue) for raw_issue in NEGATIVE_ISSUE_PATTERN.findall(raw)])
    issues = tuple(issue for issue in _extract_issues(raw) if issue not in set(negative_issues))
    dates = _extract_dates(raw)
    negative_dates = _unique(NEGATIVE_DATE_PATTERN.findall(normalized))
    negative_titles = _unique([_normalize_title(item) for item in NEGATIVE_TITLE_PATTERN.findall(raw)])
    negative_terms = _extract_negative_terms(raw)
    excludes_base = "不要基础版" in raw
    excludes_dated = "不要带日期后缀" in raw or "不要带日期" in raw
    wants_base = "基础版" in raw and not excludes_base
    wants_current = any(token in raw for token in ("现行版", "现行", "最新版", "最新版本"))
    parallel = _contains_parallel_intent(raw)
    doc_type_terms = _unique([term for term in DOC_TYPE_HINTS if term in raw])
    article_terms = _unique([f"第{_normalize_issue(num)}{suffix}" for num, suffix in ARTICLE_PATTERN.findall(raw)])
    case_numbers = _unique([number for number in CASE_NUMBER_PATTERN.findall(raw)])
    route = _infer_route(raw, titles, years, issues, dates, negative_terms)
    if route not in {"gazette_issue_lookup"} and article_terms and (titles or subject_terms):
        route = "law_article_lookup"

    target_doc_keys: list[str] = []
    if "公报" in raw and years and issues:
        for year in years:
            for issue in issues:
                target_doc_keys.append(f"{year}_{issue}_npc")

    expected_doc_count = 1
    if target_doc_keys:
        expected_doc_count = max(1, len(target_doc_keys))
    elif parallel:
        if route == "version_diff_lookup" and dates:
            expected_doc_count = 2
        elif len(titles) >= 2:
            expected_doc_count = len(titles)
        elif len(dates) >= 2:
            expected_doc_count = len(dates)
        else:
            expected_doc_count = 2

    return LegalQuerySlots(
        raw_query=raw,
        normalized_query=normalized,
        route=route,
        doc_type_terms=doc_type_terms,
        title_terms=titles,
        subject_terms=subject_terms,
        years=years,
        issue_numbers=issues,
        version_dates=dates,
        article_terms=article_terms,
        case_numbers=case_numbers,
        negative_terms=negative_terms,
        negative_title_terms=negative_titles,
        negative_issue_numbers=negative_issues,
        negative_version_dates=negative_dates,
        wants_base_version=wants_base,
        wants_current_version=wants_current,
        excludes_base_version=excludes_base,
        excludes_dated_version=excludes_dated,
        parallel=parallel,
        expected_doc_count=expected_doc_count,
        target_doc_keys=_unique(target_doc_keys),
    )


def document_version_date(document_name: str) -> str:
    normalized = normalize_digits(document_name or "")
    match = DATE8_PATTERN.search(normalized)
    return match.group(1) if match else ""


def document_is_dated_version(document_name: str) -> bool:
    return bool(document_version_date(document_name))


def _document_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(chunk.get(key, "") or "")
        for key in (
            "document_name",
            "doc_type",
            "authority",
            "legal_domain",
            "effective_or_revision_date",
            "version_scope",
            "article_anchors",
            "case_anchors",
            "content_with_weight",
        )
    )


def _matches_title(doc_norm: str, title: str, all_titles: tuple[str, ...]) -> bool:
    title_norm = normalize_match_text(title)
    if not title_norm or title_norm not in doc_norm:
        return False
    for other in all_titles:
        other_norm = normalize_match_text(other)
        if other_norm != title_norm and title_norm in other_norm and other_norm in doc_norm:
            return False
    return True


def matched_title_count(document_name: str, slots: LegalQuerySlots) -> int:
    doc_norm = normalize_match_text(document_name)
    return sum(1 for title in slots.title_terms if _matches_title(doc_norm, title, slots.title_terms))


def document_conflicts_slots(document_name: str, slots: LegalQuerySlots) -> bool:
    doc_norm = normalize_match_text(document_name)
    version_date = document_version_date(document_name)
    positive_title_hit = matched_title_count(document_name, slots) > 0

    if slots.negative_issue_numbers and "公报" in document_name:
        for issue in slots.negative_issue_numbers:
            if f"_{issue}_npc" in normalize_digits(document_name) or f"_{issue}." in normalize_digits(document_name):
                return True

    if slots.excludes_dated_version and version_date:
        return True
    if slots.excludes_base_version and not version_date:
        return True
    if slots.negative_version_dates and version_date in set(slots.negative_version_dates):
        return True

    for title in slots.negative_title_terms:
        title_norm = normalize_match_text(title)
        if title_norm and title_norm in doc_norm and not positive_title_hit:
            return True

    if "基础版" in slots.negative_terms and not version_date:
        return True

    return False


def document_matches_positive_anchor(document_name: str, slots: LegalQuerySlots) -> bool:
    if slots.target_doc_keys:
        name_match = normalize_match_text(normalize_digits(document_name))
        return any(key in name_match for key in slots.target_doc_keys)
    if slots.title_terms and matched_title_count(document_name, slots) <= 0:
        return False
    if slots.subject_terms:
        name_match = normalize_match_text(document_name)
        return any(normalize_match_text(term) in name_match for term in slots.subject_terms)
    return True


def legal_metadata_score(chunk: dict[str, Any], slots: LegalQuerySlots) -> int:
    if not slots.strong_constraint:
        return 0

    document_name = str(chunk.get("document_name", "") or chunk.get("docnm_kwd", ""))
    full_text = _document_text(chunk)
    doc_norm = normalize_match_text(document_name)
    text_norm = normalize_match_text(full_text)
    normalized_name = normalize_digits(document_name)
    score = 0

    if document_conflicts_slots(document_name, slots):
        score -= 200

    if slots.title_terms:
        title_hits = matched_title_count(document_name, slots)
        score += title_hits * 35
        if title_hits == len(slots.title_terms):
            score += 20
        if slots.route == "law_article_lookup" and title_hits:
            score += 70
            if not any(token in document_name for token in ("关于适用", "解释", "批复", "案例", "合同")):
                score += 45
            else:
                score -= 55

    for subject in slots.subject_terms:
        subject_norm = normalize_match_text(subject)
        if subject_norm and subject_norm in text_norm:
            score += 18

    if "公报" in slots.doc_type_terms or slots.route == "gazette_issue_lookup":
        if "公报" in document_name:
            score += 40
        if "npc" in normalized_name.lower():
            score += 15
        for key in slots.target_doc_keys:
            if key in normalize_match_text(normalized_name):
                score += 160
        for year in slots.years:
            if year in normalized_name:
                score += 25
        for issue in slots.issue_numbers:
            if f"_{issue}_npc" in normalized_name or f"_{issue}." in normalized_name:
                score += 35

    version_date = document_version_date(document_name)
    for date in slots.version_dates:
        if date == version_date or date in text_norm:
            score += 55
        elif version_date:
            score -= 15

    if slots.wants_current_version and version_date:
        score += 35
    if slots.wants_base_version:
        score += 60 if not version_date else -45

    for article in slots.article_terms:
        if normalize_match_text(article) in text_norm:
            score += 15
        elif slots.route == "law_article_lookup":
            score -= 10
    for case_number in slots.case_numbers:
        if case_number in text_norm:
            score += 35

    return score


def legal_hard_filter(chunks: list[dict[str, Any]], slots: LegalQuerySlots) -> list[dict[str, Any]]:
    if not slots.strong_constraint or not chunks:
        return chunks

    no_conflicts = [chunk for chunk in chunks if not document_conflicts_slots(str(chunk.get("document_name", "")), slots)]
    if no_conflicts:
        chunks = no_conflicts

    if slots.route == "gazette_issue_lookup" or (slots.target_doc_keys and "公报" in slots.raw_query):
        exact = [
            chunk
            for chunk in chunks
            if any(key in normalize_match_text(normalize_digits(str(chunk.get("document_name", "")))) for key in slots.target_doc_keys)
        ]
        if exact:
            return exact + [chunk for chunk in chunks if chunk not in exact]

    if slots.route == "law_article_lookup" and slots.title_terms:
        title_matched = [
            chunk
            for chunk in chunks
            if matched_title_count(str(chunk.get("document_name", "") or chunk.get("docnm_kwd", "")), slots) > 0
        ]
        if title_matched:
            return title_matched + [chunk for chunk in chunks if chunk not in title_matched]

    if slots.version_dates or slots.wants_base_version or slots.excludes_dated_version:
        matched = [chunk for chunk in chunks if legal_metadata_score(chunk, slots) > 0]
        if matched:
            return matched + [chunk for chunk in chunks if chunk not in matched]

    return chunks


def target_coverage_key(document_name: str, slots: LegalQuerySlots) -> str:
    normalized_name = normalize_digits(document_name or "")
    name_match = normalize_match_text(normalized_name)
    for key in slots.target_doc_keys:
        if key in name_match:
            return key

    for title in sorted(slots.title_terms, key=len, reverse=True):
        title_norm = normalize_match_text(title)
        if title_norm and title_norm in name_match:
            version_date = document_version_date(document_name)
            return f"title:{title_norm}:{version_date or 'base'}"

    for subject in slots.subject_terms:
        subject_norm = normalize_match_text(subject)
        if subject_norm and subject_norm in name_match:
            version_date = document_version_date(document_name)
            return f"subject:{subject_norm}:{version_date or 'base'}"

    return normalize_match_text(document_name)


def coverage_aware_rerank(chunks: list[dict[str, Any]], slots: LegalQuerySlots, limit: int) -> list[dict[str, Any]]:
    if slots.expected_doc_count <= 1 or not chunks:
        return chunks[:limit]

    selected: list[dict[str, Any]] = []
    seen_coverage: set[str] = set()
    for chunk in chunks:
        key = target_coverage_key(str(chunk.get("document_name", "")), slots)
        if key and key not in seen_coverage:
            selected.append(chunk)
            seen_coverage.add(key)
        if len(seen_coverage) >= slots.expected_doc_count:
            break

    for chunk in chunks:
        if len(selected) >= limit:
            break
        if chunk in selected:
            continue
        selected.append(chunk)

    return selected[:limit]


def rank_corpus_documents(list_file: Path, slots: LegalQuerySlots, limit: int) -> list[dict[str, Any]]:
    if not slots.strong_constraint or not list_file.exists():
        return []

    try:
        raw_lines = list_file.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []

    ranked: list[tuple[int, int, Path, dict[str, Any]]] = []
    for order, raw_line in enumerate(raw_lines):
        raw_line = (raw_line or "").strip().strip('"').strip("'")
        if not raw_line:
            continue
        pdf_path = Path(raw_line.replace("\\", "/"))
        file_name = pdf_path.name
        if not file_name.lower().endswith(".pdf"):
            continue
        pseudo_chunk = {
            "document_name": file_name,
            "content_with_weight": f"{Path(file_name).stem} {'基础版' if not document_is_dated_version(file_name) else document_version_date(file_name)}",
        }
        score = legal_metadata_score(pseudo_chunk, slots)
        if (
            score <= 0
            or document_conflicts_slots(file_name, slots)
            or not document_matches_positive_anchor(file_name, slots)
        ):
            continue
        ranked.append((score, -order, pdf_path, pseudo_chunk))

    ranked.sort(key=lambda item: (item[0], item[1], item[2].name), reverse=True)
    results: list[dict[str, Any]] = []
    for score, _, pdf_path, pseudo_chunk in ranked[: max(limit, slots.expected_doc_count)]:
        file_name = pdf_path.name
        results.append(
            {
                "file_name": file_name,
                "doc_id": Path(file_name).stem,
                "content": pseudo_chunk["content_with_weight"],
                "score": score,
            }
        )
    return results[:limit]
