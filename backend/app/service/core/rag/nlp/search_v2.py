#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
import re
from dataclasses import dataclass

from service.core.rag.settings import TAG_FLD, PAGERANK_FLD
from service.core.retrieval_intent import (
    QueryIntentProfile,
    classify_query_intent,
    fusion_weights_for_profile,
    rerank_vector_weight_for_profile,
)
from service.core.rag.utils import rmSpace
from service.core.rag.nlp import rag_tokenizer, query
import numpy as np
from service.core.rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr
from service.core.rag.nlp.model import generate_embedding, rerank_similarity
from service.core.rag_config import (
    DENSE_SIMILARITY_FALLBACK,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_USE_INTENT_BONUS,
    DEFAULT_USE_MODEL_RERANK,
    FUSION_WEIGHT_TEXT,
    FUSION_WEIGHT_VECTOR,
    RERANK_CANDIDATE_CAP,
    RERANK_PAGE_LIMIT,
    RERANK_TOKEN_CAP,
)

def index_name(uid): return f"{uid}"

REPORT_PERIOD_PATTERN = re.compile(r"\d{4}(?:年报摘要|中报摘要|年中报|年报|三季报|中报)")


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        qv = generate_embedding(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    @staticmethod
    def _query_profile(query_text: str) -> QueryIntentProfile:
        return classify_query_intent(query_text)

    @staticmethod
    def _metadata_weights(profile: QueryIntentProfile) -> dict[str, float]:
        weights = {
            "company_kwd": 0.14,
            "report_period_kwd": 0.11,
            "report_type_kwd": 0.06,
            "source_kwd": 0.04,
        }
        if profile.needs_exact_match:
            weights["company_kwd"] += 0.04
            weights["report_period_kwd"] += 0.05
        if profile.wants_actual_announcement:
            weights["report_period_kwd"] += 0.03
            weights["report_type_kwd"] += 0.06
        if profile.prefers_announcement:
            weights["report_period_kwd"] += 0.02
            weights["report_type_kwd"] += 0.04
        if profile.prefers_commentary:
            weights["report_type_kwd"] += 0.04
            weights["source_kwd"] += 0.04
        if profile.prefers_numeric_commentary:
            weights["report_type_kwd"] += 0.04
            weights["source_kwd"] += 0.03
        if profile.is_comparison:
            weights["report_period_kwd"] += 0.03
        return weights

    @staticmethod
    def _metadata_bonus(query_text: str, chunk: dict, profile: QueryIntentProfile | None = None) -> float:
        profile = profile or classify_query_intent(query_text)
        lowered_query = query_text.lower()
        bonus = 0.0

        for field_name, weight in Dealer._metadata_weights(profile).items():
            raw_value = chunk.get(field_name, "")
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                text = str(value).strip()
                if text and text.lower() in lowered_query:
                    bonus += weight
                    break

        for token in ("营收", "收入", "归母", "净利润", "毛利率", "销量", "费用率", "现金流"):
            if token in query_text and token in chunk.get("content_with_weight", ""):
                bonus += 0.03

        return bonus

    @staticmethod
    def _extract_period_terms(query_text: str) -> list[str]:
        terms: list[str] = []
        for pattern in (
            r"\d{2}(?:Q[1-4](?:~[1-4])?|Q1~3|H[12])",
            r"\d{4}(?:年报|三季报|中报|半年报)",
        ):
            for term in re.findall(pattern, query_text, flags=re.IGNORECASE):
                if term not in terms:
                    terms.append(term)
        return terms

    @staticmethod
    def _numeric_fact_count(content: str) -> int:
        patterns = (
            r"\d+(?:\.\d+)?\s*(?:亿元|亿|万元|万)",
            r"(?:同比|环比)?\s*[+\-]?\d+(?:\.\d+)?%",
            r"\d+(?:\.\d+)?\s*(?:gwh|wh|pct)",
        )
        return sum(1 for pattern in patterns if re.search(pattern, content, flags=re.IGNORECASE))

    @classmethod
    def _query_content_bonus(
        cls,
        query_text: str,
        chunk: dict,
        profile: QueryIntentProfile | None = None,
    ) -> float:
        profile = profile or classify_query_intent(query_text)
        content = chunk.get("content_with_weight", "")
        if not content:
            return 0.0

        bonus = 0.0
        query_needs_amount = profile.is_numeric

        period_matches = sum(1 for term in cls._extract_period_terms(query_text) if term in content)
        bonus += min(0.12, period_matches * 0.06)

        metric_groups = (
            (("营收", "收入", "收现"), 0.07),
            (("归母", "净利润", "净利", "扣非"), 0.07),
            (("毛利率", "净利率", "费用率"), 0.06),
            (("动力电池", "储能", "材料回收"), 0.08),
        )
        for keywords, weight in metric_groups:
            if any(keyword in query_text for keyword in keywords) and any(keyword in content for keyword in keywords):
                bonus += weight

        if query_needs_amount and re.search(r"\d+(?:\.\d+)?\s*(?:亿元|亿|万元|万|gwh|wh)", content, flags=re.IGNORECASE):
            bonus += 0.05
        if "同比" in query_text and re.search(r"同比\s*[+\-]?\d+(?:\.\d+)?%|同比[+\-]?\d+(?:\.\d+)?%", content):
            bonus += 0.05
        if "环比" in query_text and re.search(r"环比\s*[+\-]?\d+(?:\.\d+)?%|环比[+\-]?\d+(?:\.\d+)?%", content):
            bonus += 0.04
        if profile.is_risk and re.search(r"风险|承压|波动|不及预期|压力|扰动", content):
            bonus += 0.12
        if query_needs_amount and cls._numeric_fact_count(content) >= 2:
            bonus += 0.04
        if "分别" in query_text and cls._numeric_fact_count(content) >= 3:
            bonus += 0.04
        if re.search(r"相关研究|证券分析师|执业证书", content) and cls._numeric_fact_count(content) == 0:
            bonus -= 0.03
        if profile.is_risk:
            risk_signal_hits = re.findall(
                r"\u98ce\u9669\u63d0\u793a|\u98ce\u9669\u56e0\u7d20|\u4e0d\u53ca\u9884\u671f|\u7ade\u4e89\u52a0\u5267|\u627f\u538b|\u6ce2\u52a8|\u6270\u52a8|\u4e0b\u6ed1|\u4e0d\u786e\u5b9a\u6027",
                content,
            )
            if re.search(r"\u98ce\u9669\u63d0\u793a", content):
                bonus += 0.22
            if re.search(r"(?:\u98ce\u9669\u63d0\u793a|\u98ce\u9669\u56e0\u7d20)\s*[::\uff1a]", content):
                bonus += 0.08
            bonus += min(0.12, len(risk_signal_hits) * 0.03)
            if re.search(
                r"(?:\u9500\u91cf|\u9700\u6c42|\u4ef7\u683c|\u539f\u6750\u6599|\u7ade\u4e89|\u76c8\u5229|\u6bdb\u5229\u7387).{0,8}(?:\u4e0d\u53ca\u9884\u671f|\u6ce2\u52a8|\u627f\u538b|\u4e0b\u6ed1)",
                content,
            ):
                bonus += 0.06
            if cls._numeric_fact_count(content) >= 2:
                bonus -= 0.05
            if re.search(r"\u76f8\u5173\u7814\u7a76|\u8bc1\u5238\u5206\u6790\u5e08|\u6267\u4e1a\u8bc1\u4e66", content):
                bonus -= 0.08

        if any(token in query_text for token in ("现金流", "现金流量", "经营活动")):
            if re.search(r"经营活动.*现金流|现金流量净额|现金流净额", content):
                bonus += 0.14
            if cls._numeric_fact_count(content) >= 1:
                bonus += 0.02

        if any(token in query_text for token in ("分红", "派现", "利润分配", "现金红利")):
            if re.search(r"每股现金分红|派发现金红利|现金红利|利润分配", content):
                bonus += 0.16
            if cls._numeric_fact_count(content) >= 1:
                bonus += 0.02

        if profile.prefers_numeric_commentary:
            if re.search(r"(实现|录得|达到).{0,10}(营业收入|营收|归母净利润|净利润)", content):
                bonus += 0.08
            if cls._numeric_fact_count(content) >= 2 and any(metric in content for metric in profile.metric_terms):
                bonus += 0.05
            if re.search(r"相关研究|证券分析师|执业证书", content) and cls._numeric_fact_count(content) == 0:
                bonus -= 0.05

        if profile.is_table and "<table" in content.lower():
            bonus += 0.10
        if profile.wants_actual_announcement and re.search(r"本报告期|公告", content):
            bonus += 0.05

        return bonus

    @staticmethod
    def _extract_company_period(query_text: str) -> tuple[str, str]:
        normalized = re.sub(r"\s+", "", query_text)
        match = REPORT_PERIOD_PATTERN.search(normalized)
        if not match:
            return "", ""

        period = match.group(0)
        company = normalized[:match.start()]
        company = re.sub(
            r"(点评里的|研报|解读|怎么看|怎么解读|有哪些|是什么|多少|财报|公告|资料|核心内容|风险提示|风险点)+$",
            "",
            company,
        )
        return company, period

    @classmethod
    def _extract_broker(cls, query_text: str) -> str:
        normalized = re.sub(r"\s+", "", query_text)
        company, period = cls._extract_company_period(normalized)
        remainder = normalized
        if company and remainder.startswith(company):
            remainder = remainder[len(company):]
        if period:
            remainder = remainder.replace(period, "", 1)

        broker_match = re.search(r"[\u4e00-\u9fa5]{2,10}证券", remainder)
        if broker_match:
            return broker_match.group(0)
        return ""

    @classmethod
    def _wants_actual_announcement(cls, query_text: str) -> bool:
        if cls._extract_broker(query_text):
            return False
        if any(token in query_text for token in ("点评", "研报", "解读", "风险")):
            return False
        if any(
            token in query_text
            for token in (
                "公告",
                "原文",
                "全文",
                "正式披露",
                "官方披露",
                "公司披露",
                "公司自己披露",
                "披露文件",
                "完整年报",
                "完整中报",
                "完整季报",
                "完整公告",
                "先看公告",
                "先看公司披露",
                "不要券商解读",
                "不看研报",
                "不看券商",
                "本报告期",
            )
        ):
            return True
        if any(token in query_text for token in ("分红", "派息", "利润分配", "现金红利", "每10股")):
            return True
        return False

    @classmethod
    def _doc_name_bonus(
        cls,
        query_text: str,
        chunk: dict,
        profile: QueryIntentProfile | None = None,
    ) -> float:
        profile = profile or classify_query_intent(query_text)
        doc_name = str(chunk.get("docnm_kwd", "")).split("/")[-1]
        content = str(chunk.get("content_with_weight", ""))
        title = " ".join(cls._token_list(chunk.get("title_tks", "")))
        company, period = cls._extract_company_period(query_text)
        if not company:
            company = profile.company

        bonus = 0.0
        if company:
            if company in doc_name:
                bonus += 0.45
            else:
                bonus -= 0.18

        if period:
            if period in doc_name:
                bonus += 0.30
            else:
                bonus -= 0.12

        wants_commentary = profile.prefers_commentary
        wants_announcement = profile.prefers_announcement
        wants_risk = profile.is_risk
        wants_summary = profile.prefers_summary or "摘要" in query_text
        wants_table = profile.is_table
        wants_actual_announcement = profile.wants_actual_announcement or cls._wants_actual_announcement(query_text)

        if wants_summary:
            if "摘要" in doc_name:
                bonus += 0.34
            if "公告" in doc_name and not wants_actual_announcement:
                bonus -= 0.08
            if "点评" in doc_name:
                bonus -= 0.12

        if wants_commentary:
            if "点评" in doc_name:
                bonus += 0.35
            if "公告" in doc_name:
                bonus -= 0.22
            if "摘要" in doc_name:
                bonus -= 0.08

        if wants_announcement:
            if "公告" in doc_name:
                bonus += 0.42
            if "点评" in doc_name:
                bonus -= 0.18
            if "摘要" in doc_name and not wants_summary:
                bonus -= 0.10

        if wants_actual_announcement:
            if period:
                if period in doc_name:
                    bonus += 0.12
                else:
                    bonus -= 0.24
            if "公告" in doc_name:
                bonus += 0.28
            if "摘要" in doc_name:
                bonus -= 0.08
            if "点评" in doc_name:
                bonus -= 0.18
            if not any(token in doc_name for token in ("公告", "摘要", "点评")):
                bonus -= 0.10

        if profile.prefers_numeric_commentary:
            if "点评" in doc_name:
                bonus += 0.28
            if "公告" in doc_name:
                bonus -= 0.20
            if "摘要" in doc_name:
                bonus -= 0.08
            if "表格" in doc_name and not wants_table:
                bonus -= 0.08

        if wants_summary and "摘要" in doc_name:
            bonus += 0.24

        if wants_table and "表格" in doc_name:
            bonus += 0.24

        if wants_risk:
            if "点评" in doc_name:
                bonus += 0.18
            if "公告" in doc_name:
                bonus -= 0.12
            if re.search(r"风险提示|风险|承压|不及预期|竞争加剧", f"{title} {content}"):
                bonus += 0.18

        broker = cls._extract_broker(query_text)
        if broker:
            same_report_family = bool(
                company and company in doc_name and period and period in doc_name and "点评" in doc_name
            )
            if broker in doc_name:
                bonus += 0.30
                if wants_commentary or wants_risk:
                    bonus += 0.12
                    if same_report_family:
                        bonus += 0.18
            else:
                bonus -= 0.14
                if wants_commentary or wants_risk:
                    bonus -= 0.08
                    if same_report_family:
                        bonus -= 0.12

        return bonus

    @classmethod
    def _intent_bonus(
        cls,
        query_text: str,
        chunk: dict,
        profile: QueryIntentProfile | None = None,
    ) -> float:
        return (
            cls._metadata_bonus(query_text, chunk, profile)
            + cls._query_content_bonus(query_text, chunk, profile)
            + cls._doc_name_bonus(query_text, chunk, profile)
        )

    @classmethod
    def _broker_candidate_adjustments(
        cls,
        query_text: str,
        chunks: list[dict],
        profile: QueryIntentProfile | None = None,
    ) -> np.ndarray:
        profile = profile or classify_query_intent(query_text)
        broker = profile.brokers[0] if len(profile.brokers) == 1 else cls._extract_broker(query_text)
        if not broker:
            return np.zeros(len(chunks), dtype=float)

        wants_commentary = profile.prefers_commentary
        wants_risk = profile.is_risk
        if not (wants_commentary or wants_risk):
            return np.zeros(len(chunks), dtype=float)

        company, period = cls._extract_company_period(query_text)
        if not company:
            company = profile.company
        same_family_flags: list[bool] = []
        has_matching_family = False

        for chunk in chunks:
            doc_name = str(chunk.get("docnm_kwd", "")).split("/")[-1]
            same_family = bool(
                company and company in doc_name and period and period in doc_name and "点评" in doc_name
            )
            same_family_flags.append(same_family)
            if same_family and broker in doc_name:
                has_matching_family = True

        if not has_matching_family:
            return np.zeros(len(chunks), dtype=float)

        adjustments = []
        for chunk, same_family in zip(chunks, same_family_flags):
            doc_name = str(chunk.get("docnm_kwd", "")).split("/")[-1]
            if not same_family:
                adjustments.append(0.0)
            elif broker in doc_name:
                adjustments.append(0.06)
            else:
                adjustments.append(-0.06)
        return np.array(adjustments, dtype=float)

    @staticmethod
    def _expand_query_for_recall(query_text: str) -> str:
        normalized = str(query_text).strip()
        additions: list[str] = []

        if "风险" in normalized:
            additions.extend(["点评", "风险提示"])
        elif "研报" in normalized and "公告" not in normalized:
            additions.append("点评")
        elif "点评" in normalized and "公告" not in normalized:
            additions.append("点评")

        for token in ("点评", "研报", "风险提示"):
            if token in normalized and token in additions:
                additions = [item for item in additions if item != token]

        if not additions:
            return normalized
        return f"{normalized} {' '.join(additions)}"

    def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: dict | None = None,
               query_profile: QueryIntentProfile | None = None
               ):
        filters = self.get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "company_kwd", "report_period_kwd", "report_type_kwd", "source_kwd",
                       "table_dense_int", "table_headers_kwd", "table_rows_kwd",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            highlightFields = ["content_ltks", "title_tks"] if highlight else []
            matchText, keywords = self.qryr.question(qst, min_match=0.3)
            # if emb_mdl is None:
            #     matchExprs = [matchText]
            #     res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
            #                                 idx_names, kb_ids, rank_feature=rank_feature)
            #     total = self.dataStore.getTotal(res)
            #     logging.debug("Dealer.search TOTAL: {}".format(total))
            # else:
            matchDense = self.get_vector(qst, emb_mdl, topk, req.get("similarity", DEFAULT_SIMILARITY_THRESHOLD))
            q_vec = matchDense.embedding_data
            src.append(f"q_{len(q_vec)}_vec")
            query_profile = query_profile or self._query_profile(qst)
            text_weight, vector_weight = fusion_weights_for_profile(query_profile)
            fusionExpr = FusionExpr(
                "weighted_sum",
                topk,
                {"weights": f"{text_weight:.2f}, {vector_weight:.2f}"},
            )
            matchExprs = [matchText, matchDense, fusionExpr]
            res = self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                        idx_names, kb_ids, rank_feature=rank_feature)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
            # If result is empty, try again with lower min_match
            if total == 0:
                matchText, _ = self.qryr.question(qst, min_match=0.1)
                filters.pop("doc_ids", None)
                matchDense.extra_options["similarity"] = DENSE_SIMILARITY_FALLBACK
                res = self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                            orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:
                        continue
                    if kk in kwds:
                        continue
                    kwds.add(kk)

        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.getChunkIds(res)
        keywords = list(kwds)
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.getFields(res, src),
            keywords=keywords
        )

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0]*len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor/np.sqrt(denor)/q_denor)
        return np.array(rank_fea)*10. + pageranks

    @staticmethod
    def _token_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [token for token in value.split() if token]
        if isinstance(value, (list, tuple, set)):
            tokens = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    tokens.extend(token for token in item.split() if token)
                else:
                    tokens.append(str(item))
            return tokens
        return [token for token in str(value).split() if token]

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None,
               use_intent_bonus: bool = True,
               query_profile: QueryIntentProfile | None = None
               ):
        query_profile = query_profile or self._query_profile(query)
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = self._token_list(sres.field[i].get(cfield, ""))
            title_tks = self._token_list(sres.field[i].get("title_tks", ""))
            question_tks = self._token_list(sres.field[i].get("question_tks", ""))
            important_kwd = self._token_list(sres.field[i].get("important_kwd", []))
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)
        if use_intent_bonus:
            fields = [sres.field[i] for i in sres.ids]
            intent_bonus = np.array([self._intent_bonus(query, field, query_profile) for field in fields], dtype=float)
            intent_bonus += self._broker_candidate_adjustments(query, fields, query_profile)
        else:
            intent_bonus = np.zeros(len(sres.ids), dtype=float)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea + intent_bonus, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None,
                        use_intent_bonus: bool = True,
                        query_profile: QueryIntentProfile | None = None):
        query_profile = query_profile or self._query_profile(query)
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = self._token_list(sres.field[i].get(cfield, ""))
            title_tks = self._token_list(sres.field[i].get("title_tks", ""))
            important_kwd = self._token_list(sres.field[i].get("important_kwd", []))
            # Keep the rerank payload focused on the strongest signals instead of
            # sending the full chunk token stream for every candidate.
            tks = (important_kwd + title_tks + content_ltks)[:RERANK_TOKEN_CAP]
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)
        if use_intent_bonus:
            fields = [sres.field[i] for i in sres.ids]
            intent_bonus = np.array([self._intent_bonus(query, field, query_profile) for field in fields], dtype=float)
            intent_bonus += self._broker_candidate_adjustments(query, fields, query_profile)
        else:
            intent_bonus = np.zeros(len(sres.ids), dtype=float)

        return tkweight * (np.array(tksim) + rank_fea) + vtweight * vtsim + intent_bonus, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: dict | None = {PAGERANK_FLD: 10},
                  use_model_rerank: bool = DEFAULT_USE_MODEL_RERANK,
                  use_intent_bonus: bool = DEFAULT_USE_INTENT_BONUS):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        query_profile = self._query_profile(question)
        vector_similarity_weight = rerank_vector_weight_for_profile(query_profile, vector_similarity_weight)

        search_question = self._expand_query_for_recall(question)
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "size": max(page_size * RERANK_PAGE_LIMIT, RERANK_CANDIDATE_CAP),
               "question": search_question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}

        if page > RERANK_PAGE_LIMIT:
            req["page"] = page
            req["size"] = page_size

        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                           kb_ids, embd_mdl, highlight, rank_feature=rank_feature, query_profile=query_profile)
        ranks["total"] = sres.total


        if page <= RERANK_PAGE_LIMIT:
            if sres.total > 0:
                logging.debug("Running rerank stage")
                if use_model_rerank:
                    try:
                        sim, tsim, vsim = self.rerank_by_model(
                            rerank_mdl,
                            sres,
                            question,
                            1 - vector_similarity_weight,
                            vector_similarity_weight,
                            rank_feature=rank_feature,
                            use_intent_bonus=use_intent_bonus,
                            query_profile=query_profile,
                        )
                    except Exception as exc:
                        logging.warning(
                            "rerank_by_model failed; fallback to heuristic rerank: %s: %s",
                            exc.__class__.__name__,
                            exc,
                        )
                        sim, tsim, vsim = self.rerank(
                            sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                            rank_feature=rank_feature,
                            use_intent_bonus=use_intent_bonus,
                            query_profile=query_profile,
                        )
                else:
                    sim, tsim, vsim = self.rerank(
                        sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                        rank_feature=rank_feature,
                        use_intent_bonus=use_intent_bonus,
                        query_profile=query_profile,
                    )
            else:
                sim, tsim, vsim = self.rerank(
                    sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                    rank_feature=rank_feature,
                    use_intent_bonus=use_intent_bonus,
                    query_profile=query_profile)
            idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]
        else:
            sim = tsim = vsim = [1] * len(sres.ids)
            idx = list(range(len(sres.ids)))

        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")
            did = chunk.get("doc_id", "")
            position_int = chunk.get("position_int", [])
            display_rank = len(ranks["chunks"]) + 1
            source_rank = int(i) + 1
            d = {
                "chunk_id": id,
                "id": id,
                "display_rank": display_rank,
                "rank": source_rank,
                "source_rank": source_rank,
                "content_ltks": chunk["content_ltks"],
                "content_with_weight": chunk["content_with_weight"],
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),
                "question_kwd": chunk.get("question_kwd", []),
                "question_tks": chunk.get("question_tks", ""),
                "company_kwd": chunk.get("company_kwd", ""),
                "report_period_kwd": chunk.get("report_period_kwd", ""),
                "report_type_kwd": chunk.get("report_type_kwd", ""),
                "source_kwd": chunk.get("source_kwd", ""),
                "table_dense_int": chunk.get("table_dense_int", 0),
                "table_headers_kwd": chunk.get("table_headers_kwd", []),
                "table_rows_kwd": chunk.get("table_rows_kwd", []),
                "image_id": chunk.get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": chunk.get(vector_column, zero_vector),
                "positions": position_int,
                "page_num_int": chunk.get("page_num_int", []),
            }
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                                                       v in sorted(ranks["doc_aggs"].items(),
                                                                   key=lambda x: x[1]["count"] * -1)]
        ranks["chunks"] = ranks["chunks"][:page_size]

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        condition = {"doc_id": doc_id}
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.getFields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            if len(dict_chunks.values()) < bs:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.getAggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.getAggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a: c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a: max(1, c) for a, c in tag_fea}
