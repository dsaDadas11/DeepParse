from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from sqlalchemy import text
from sqlalchemy.orm import Session

from runtime_config import REWRITE_API_KEY, REWRITE_BASE_URL, REWRITE_MODEL

MAX_HISTORY_TURNS = 3
MAX_HISTORY_QUESTION_CHARS = 200
MAX_HISTORY_ANSWER_CHARS = 280
BROKER_ALIASES = (
    "东吴",
    "国元",
    "东兴",
    "华泰",
    "华创",
    "民生",
    "中信",
    "中金",
    "国信",
    "海通",
    "申万",
    "广发",
    "招商",
    "光大",
    "平安",
    "兴业",
    "长江",
    "国联",
    "国金",
    "西部",
)
BROKER_FALSE_POSITIVE_SUFFIXES = ("银行",)
METRIC_KEYWORDS = (
    "归母净利润",
    "净利润",
    "利润预测",
    "盈利预测",
    "营业总收入",
    "营业收入",
    "营收",
    "净息差",
    "不良贷款率",
    "拨备覆盖率",
    "核心一级资本充足率",
    "摘要",
)
FORECAST_HINTS = ("预测", "怎么看", "看多少", "看到了", "那组", "怎么调", "盈利")
FOLLOWUP_PREFIX_PATTERN = re.compile(r"^(先看|再看|看下|看一下|先问|先说|先聊|帮我看下|帮我看一下|麻烦看下)")
FOLLOWUP_FILLER_PATTERN = re.compile(r"^(那|那么|这|这个|对应|还有|另外|然后|再|其中|它|他|她)+")
PERIOD_PATTERN = re.compile(
    r"(20\d{2}年(?:年报|中报摘要|中报|半年报|三季报|一季报|摘要)"
    r"|20\d{2}(?:Q1~3|Q1-Q3|Q[1-4]|H1|H2)"
    r"|\d{2}(?:Q1~3|Q1-Q3|Q[1-4]|H1|H2)"
    r"|20\d{2}(?:年报|年中报摘要|年中报|年半年报|年三季报|年一季报|年摘要)"
    r"|20\d{2}年"
    r"|20\d{2}-20\d{2}年"
    r"|\d{2}-\d{2}年)"
)
REPORT_TYPE_PATTERN = re.compile(r"(年报|中报摘要|中报|半年报|三季报|一季报|摘要)")
COMPANY_STOP_PHRASES = ("不同券商", "各家券商", "券商对")


def create_rewrite_client() -> OpenAI:
    return OpenAI(
        api_key=REWRITE_API_KEY,
        base_url=REWRITE_BASE_URL,
    )


def strip_code_fence(content: str) -> str:
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.search(pattern, content.strip(), re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else content.strip()


def clip_text(value: str, max_chars: int) -> str:
    text = (value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def normalize_history_turns(
    history_turns: list[dict[str, Any]] | None,
    limit: int = MAX_HISTORY_TURNS,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for item in history_turns or []:
        if not isinstance(item, dict):
            continue

        user_question = clip_text(
            str(item.get("user_question") or item.get("question") or item.get("user") or ""),
            MAX_HISTORY_QUESTION_CHARS,
        )
        model_answer = clip_text(
            str(item.get("model_answer") or item.get("answer") or item.get("assistant") or ""),
            MAX_HISTORY_ANSWER_CHARS,
        )

        if not user_question and not model_answer:
            continue

        normalized.append(
            {
                "user_question": user_question,
                "model_answer": model_answer,
            }
        )

    if limit <= 0:
        return normalized
    return normalized[-limit:]


def format_history_for_prompt(history_turns: list[dict[str, Any]] | None) -> str:
    normalized = normalize_history_turns(history_turns)
    if not normalized:
        return ""

    lines: list[str] = []
    for index, turn in enumerate(normalized, start=1):
        if turn["user_question"]:
            lines.append(f"Turn {index} User: {turn['user_question']}")
        if turn["model_answer"]:
            lines.append(f"Turn {index} Assistant: {turn['model_answer']}")
    return "\n".join(lines)


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


def strip_leading_context(text: str) -> str:
    stripped = FOLLOWUP_PREFIX_PATTERN.sub("", text.strip())
    return FOLLOWUP_FILLER_PATTERN.sub("", stripped).strip(" ，。？?：:")


def extract_period_tokens(text: str) -> list[str]:
    return unique_preserve_order(PERIOD_PATTERN.findall(text or ""))


def extract_year_numbers(text: str) -> list[int]:
    years: list[int] = []
    for match in re.finditer(r"(20\d{2}|\d{2})(?=年|H1|H2|Q[1-4]|Q1~3|Q1-Q3)", text or "", re.IGNORECASE):
        raw = match.group(1)
        year = int(raw if len(raw) == 4 else f"20{raw}")
        years.append(year)
    return years


def extract_report_type_tokens(text: str) -> list[str]:
    return unique_preserve_order(REPORT_TYPE_PATTERN.findall(text or ""))


def broker_alias_is_false_positive(text: str, alias: str, start: int) -> bool:
    suffix = (text or "")[start + len(alias):]
    return any(suffix.startswith(marker) for marker in BROKER_FALSE_POSITIVE_SUFFIXES)


def extract_brokers(text: str) -> list[str]:
    hits: list[str] = []
    haystack = text or ""
    for alias in BROKER_ALIASES:
        for match in re.finditer(re.escape(alias), haystack):
            if broker_alias_is_false_positive(haystack, alias, match.start()):
                continue
            hits.append(alias)
            break
    return unique_preserve_order(hits)


def extract_metric_keywords(text: str) -> list[str]:
    hits: list[str] = []
    for keyword in METRIC_KEYWORDS:
        if keyword in (text or ""):
            hits.append(keyword)
    return unique_preserve_order(hits)


def extract_company_candidate(text: str) -> str:
    cleaned = strip_leading_context(text)
    if not cleaned:
        return ""

    stop_indexes = [len(cleaned)]
    for pattern in (PERIOD_PATTERN, REPORT_TYPE_PATTERN):
        match = pattern.search(cleaned)
        if match:
            stop_indexes.append(match.start())
    for broker in BROKER_ALIASES:
        for match in re.finditer(re.escape(broker), cleaned):
            if broker_alias_is_false_positive(cleaned, broker, match.start()):
                continue
            stop_indexes.append(match.start())
            break
    for keyword in METRIC_KEYWORDS:
        index = cleaned.find(keyword)
        if index >= 0:
            stop_indexes.append(index)
    for phrase in COMPANY_STOP_PHRASES:
        index = cleaned.find(phrase)
        if index >= 0:
            stop_indexes.append(index)

    candidate = cleaned[: min(stop_indexes)].strip(" ，。？?：:")
    candidate = re.split(r"[，。？?：:\s]", candidate, maxsplit=1)[0].strip()
    if 1 < len(candidate) <= 16:
        return candidate
    bank_match = re.match(r"([\u4e00-\u9fff]{2,8}银行)", cleaned)
    if bank_match:
        return bank_match.group(1)
    return ""


def infer_intent_tokens(
    current_question: str,
    last_user_question: str,
    last_model_answer: str,
    current_metrics: list[str],
    base_metrics: list[str],
) -> list[str]:
    combined = " ".join([current_question, last_user_question, last_model_answer])
    metric_tokens = current_metrics or base_metrics

    if any(hint in combined for hint in FORECAST_HINTS):
        if any(keyword in combined for keyword in ("归母净利润", "净利润", "利润")):
            return ["归母净利润预测"]
        return ["盈利预测"]

    if metric_tokens:
        return metric_tokens[:3]

    if "摘要" in combined:
        return ["摘要指标"]

    return []


def question_needs_rewrite(question: str) -> bool:
    cleaned = question.strip()
    if len(cleaned) <= 12:
        return True
    if FOLLOWUP_FILLER_PATTERN.match(cleaned):
        return True
    if "呢" in cleaned or "对应" in cleaned:
        return True
    return False


def heuristic_standalone_query(question: str, history_turns: list[dict[str, Any]] | None) -> str:
    normalized = normalize_history_turns(history_turns, limit=2)
    if not normalized:
        return question.strip()

    last_turn = normalized[-1]
    last_user_question = last_turn["user_question"]
    last_model_answer = last_turn["model_answer"]
    current_question = question.strip()

    company = (
        extract_company_candidate(current_question)
        or extract_company_candidate(last_user_question)
        or extract_company_candidate(last_model_answer)
    )
    current_period_tokens = extract_period_tokens(current_question)
    base_period_tokens = extract_period_tokens(last_user_question)
    current_report_tokens = extract_report_type_tokens(current_question)
    base_report_tokens = extract_report_type_tokens(last_user_question)
    period_tokens = current_period_tokens or ([] if current_report_tokens else base_period_tokens)
    report_tokens = current_report_tokens or ([] if current_period_tokens else base_report_tokens)
    if (
        "年报" in current_question
        and not current_period_tokens
        and not any(char.isdigit() for char in current_question)
    ):
        history_years = extract_year_numbers(" ".join([last_user_question, last_model_answer]))
        if history_years:
            latest_year = history_years[-1]
            if any(token in " ".join([last_user_question, last_model_answer]) for token in ("H1", "H2", "中报", "半年报", "Q1", "一季报")):
                period_tokens = [f"{latest_year - 1}年年报"]
    if period_tokens and report_tokens:
        joined_periods = " ".join(period_tokens)
        report_tokens = [token for token in report_tokens if token not in joined_periods]
    broker_tokens = extract_brokers(current_question) or extract_brokers(last_user_question)
    current_metrics = extract_metric_keywords(current_question)
    base_metrics = extract_metric_keywords(last_user_question)
    intent_tokens = infer_intent_tokens(
        current_question,
        last_user_question,
        last_model_answer,
        current_metrics,
        base_metrics,
    )

    fragments = [
        company,
        " ".join(period_tokens),
        " ".join(report_tokens),
        " ".join(broker_tokens),
        " ".join(intent_tokens),
        strip_leading_context(current_question),
    ]
    query = " ".join(unique_preserve_order([fragment for fragment in fragments if fragment]))
    return query.strip() or current_question


def fallback_standalone_query(question: str, history_turns: list[dict[str, Any]] | None) -> str:
    heuristic_query = heuristic_standalone_query(question, history_turns)
    if heuristic_query:
        return heuristic_query

    normalized = normalize_history_turns(history_turns, limit=2)
    fragments: list[str] = []
    if normalized:
        last_turn = normalized[-1]
        fragments.extend(
            fragment
            for fragment in (last_turn["user_question"], last_turn["model_answer"])
            if fragment
        )
    fragments.append(question.strip())
    return " ".join(fragment for fragment in fragments if fragment).strip()


def rewrite_question_with_history(
    question: str,
    history_turns: list[dict[str, Any]] | None,
    model_name: str | None = None,
) -> tuple[str, float]:
    normalized = normalize_history_turns(history_turns)
    if not normalized:
        return question.strip(), 0.0

    if not question_needs_rewrite(question):
        return question.strip(), 0.0

    heuristic_query = heuristic_standalone_query(question, normalized)
    if not REWRITE_API_KEY or not REWRITE_BASE_URL:
        return fallback_standalone_query(question, normalized), 0.0

    prompt = f"""
You rewrite follow-up questions into standalone retrieval queries for financial-report QA.
Use the same language as the current question.
Only carry over facts that are necessary for retrieval, such as company, report period, broker, metric, and comparison target.
Do not answer the question.
Return strict JSON with the field name standalone_query.

Conversation history:
{format_history_for_prompt(normalized)}

Current question:
{question.strip()}

Candidate standalone query:
{heuristic_query}

Output format:
{{
  "standalone_query": "..."
}}
""".strip()

    start = time.perf_counter()
    client = create_rewrite_client()
    last_error: Exception | None = None

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model=model_name or REWRITE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                stream=False,
                timeout=8,
            )
            response = completion.choices[0].message.content or ""
            payload = json.loads(strip_code_fence(response))
            standalone_query = str(payload.get("standalone_query") or "").strip()
            latency_ms = (time.perf_counter() - start) * 1000
            if standalone_query:
                return standalone_query, latency_ms
            break
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            last_error = exc
            if attempt == 1:
                break
            time.sleep(1)
        except Exception:
            break

    latency_ms = (time.perf_counter() - start) * 1000
    fallback_query = heuristic_query or fallback_standalone_query(question, normalized)
    if not fallback_query:
        fallback_query = question.strip()
    _ = last_error
    return fallback_query, latency_ms


def load_session_history(
    db: Session,
    session_id: str,
    user_id: str,
    limit: int = MAX_HISTORY_TURNS,
) -> list[dict[str, str]]:
    rows = db.execute(
        text(
            """
            SELECT m.user_question, m.model_answer
            FROM messages AS m
            INNER JOIN sessions AS s
                ON s.session_id = m.session_id
            WHERE m.session_id = :session_id
              AND s.user_id = :user_id
            ORDER BY m.created_at DESC
            LIMIT :limit
            """
        ),
        {"session_id": session_id, "user_id": user_id, "limit": limit},
    ).fetchall()

    turns = [
        {
            "user_question": row.user_question,
            "model_answer": row.model_answer,
        }
        for row in reversed(rows)
    ]
    return normalize_history_turns(turns, limit=limit)
