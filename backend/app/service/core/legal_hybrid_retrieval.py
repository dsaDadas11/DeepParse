from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import jieba
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


# -----------------------------
# Data Models
# -----------------------------


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class QueryFeatures:
    raw_query: str
    normalized_query: str
    article_refs: list[str] = field(default_factory=list)
    date_tokens: list[str] = field(default_factory=list)
    version_tokens: list[str] = field(default_factory=list)
    file_hints: list[str] = field(default_factory=list)
    must_terms: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunk: Chunk
    vector_score: float
    bm25_score: float
    fusion_score: float


@dataclass
class RerankResult:
    chunk: Chunk
    vector_score: float
    bm25_score: float
    exact_bonus: float
    version_penalty: float
    final_score: float


# -----------------------------
# Utility Helpers
# -----------------------------


FULLWIDTH_MAP = str.maketrans(
    "0123456789（）［］【】：，。；＿－",
    "0123456789()[][]:,.;_-",
)

CN_NUM_MAP = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
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

ARTICLE_PATTERN = re.compile(r"第\s*([0-9零〇一二三四五六七八九十百千]+)\s*条")
DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2}[01]\d[0-3]\d)(?!\d)")
DATE_DASH_PATTERN = re.compile(r"(20\d{2})[-./年]([01]?\d)[-./月]([0-3]?\d)")
VERSION_PATTERN = re.compile(
    r"(现行版|最新版|基础版|试行版|修订版|征求意见稿|20\d{2}[_＿]\d+|\d{4}年第\d+版|\d{4}年版)",
    re.IGNORECASE,
)
FILE_HINT_PATTERN = re.compile(r"《[^》]{2,64}》|[\w\u4e00-\u9fff]{2,64}\.(?:pdf|docx|doc|txt|md)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    out = (text or "").translate(FULLWIDTH_MAP)
    out = out.replace("\u3000", " ")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def chinese_numeral_to_int(raw: str) -> Optional[int]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)

    if len(raw) == 1 and raw in CN_NUM_MAP and CN_NUM_MAP[raw] < 10:
        return CN_NUM_MAP[raw]

    total = 0
    temp = 0
    for ch in raw:
        val = CN_NUM_MAP.get(ch)
        if val is None:
            return None
        if val >= 10:
            if temp == 0:
                temp = 1
            total += temp * val
            temp = 0
        else:
            temp = temp * 10 + val
    total += temp
    return total if total > 0 else None


def safe_tokenize(text: str) -> list[str]:
    cleaned = normalize_text(text).lower()
    if not cleaned:
        return []
    tokens = [t.strip() for t in jieba.lcut(cleaned) if t.strip()]
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def normalize_article_token(token: str) -> str:
    m = ARTICLE_PATTERN.search(token)
    if not m:
        return token
    number_raw = m.group(1)
    num = chinese_numeral_to_int(number_raw)
    if num is None:
        return token
    return f"第{num}条"


def extract_version_date(text: str) -> str:
    normalized = normalize_text(text)
    date8 = DATE_PATTERN.search(normalized)
    if date8:
        return date8.group(1)

    dm = DATE_DASH_PATTERN.search(normalized)
    if dm:
        y, m, d = dm.groups()
        return f"{int(y):04d}{int(m):02d}{int(d):02d}"
    return ""


def contains_any(haystack: str, needles: Iterable[str]) -> bool:
    h = normalize_text(haystack).lower()
    return any(normalize_text(n).lower() in h for n in needles if n)


# -----------------------------
# A. Structure-aware parser
# -----------------------------


class DocumentParser:
    """Parse legal text into structure-aware chunks with rich metadata."""

    heading_patterns = [
        re.compile(r"^第[一二三四五六七八九十百千]+章\s*.*"),
        re.compile(r"^第[一二三四五六七八九十百千]+节\s*.*"),
        re.compile(r"^[一二三四五六七八九十]+、\s*.*"),
        re.compile(r"^\([一二三四五六七八九十]+\)\s*.*"),
        re.compile(r"^[0-9]+[.、]\s*.*"),
        re.compile(r"^#{1,6}\s+.*"),
    ]

    def parse_documents(self, documents: list[dict[str, Any]]) -> list[Chunk]:
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.parse_single_document(doc))
        return all_chunks

    def parse_single_document(self, doc: dict[str, Any]) -> list[Chunk]:
        text = normalize_text(str(doc.get("text", "")))
        lines = [normalize_text(line) for line in text.splitlines() if line.strip()]

        doc_id = str(doc.get("doc_id", "")) or f"doc_{abs(hash(doc.get('source_path', text[:64])))}"
        file_name = str(doc.get("file_name", ""))
        full_title = str(doc.get("full_title", "")) or file_name
        source_path = str(doc.get("source_path", ""))

        version_date = str(doc.get("version_date", "")) or extract_version_date(f"{file_name} {text[:3000]}")
        edition_tag = str(doc.get("edition_tag", "")) or self._extract_edition_tag(f"{file_name} {text[:2000]}")

        base_meta = {
            "doc_id": doc_id,
            "file_name": file_name,
            "full_title": full_title,
            "source_path": source_path,
            "version_date": version_date,
            "edition_tag": edition_tag,
        }

        article_chunks = self._split_legal_articles(lines, base_meta)
        if article_chunks:
            parsed_chunks = article_chunks
        else:
            parsed_chunks = self._split_by_heading(lines, base_meta)

        table_chunks = self._parse_tables(doc.get("tables", []), base_meta)
        return parsed_chunks + table_chunks

    def _extract_edition_tag(self, text: str) -> str:
        m = VERSION_PATTERN.search(normalize_text(text))
        return m.group(1) if m else ""

    def _split_legal_articles(self, lines: list[str], base_meta: dict[str, Any]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_chapter = ""
        current_section = ""
        current_article = ""
        buf: list[str] = []
        article_idx = 0

        def flush_article() -> None:
            nonlocal article_idx, buf, current_article
            if not current_article or not buf:
                return
            article_idx += 1
            text = "\n".join(buf).strip()
            if not text:
                return
            section_path = ">".join([p for p in [current_chapter, current_section, current_article] if p])
            meta = dict(base_meta)
            meta["section_path"] = section_path
            meta["chunk_type"] = "article"
            chunk_id = f"{base_meta['doc_id']}::article::{article_idx}"
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta))

        for line in lines:
            if re.match(r"^第[一二三四五六七八九十百千]+章", line):
                current_chapter = line
                continue
            if re.match(r"^第[一二三四五六七八九十百千]+节", line):
                current_section = line
                continue

            article_match = ARTICLE_PATTERN.match(line)
            if article_match:
                flush_article()
                current_article = normalize_article_token(line)
                buf = [line]
            elif current_article:
                buf.append(line)

        flush_article()
        return chunks

    def _split_by_heading(self, lines: list[str], base_meta: dict[str, Any]) -> list[Chunk]:
        chunks: list[Chunk] = []
        heading_stack: list[str] = []
        buf: list[str] = []
        idx = 0

        def flush_block() -> None:
            nonlocal idx, buf
            if not buf:
                return
            text = "\n".join(buf).strip()
            if not text:
                return
            idx += 1
            meta = dict(base_meta)
            meta["section_path"] = ">".join(heading_stack)
            meta["chunk_type"] = "section"
            chunk_id = f"{base_meta['doc_id']}::section::{idx}"
            chunks.append(Chunk(chunk_id=chunk_id, text=text, metadata=meta))

        for line in lines:
            if self._is_heading(line):
                flush_block()
                heading_stack = self._update_heading_stack(heading_stack, line)
                buf = [line]
            else:
                buf.append(line)
        flush_block()
        return chunks

    def _is_heading(self, line: str) -> bool:
        return any(p.match(line) for p in self.heading_patterns)

    def _update_heading_stack(self, heading_stack: list[str], heading: str) -> list[str]:
        # Simple hierarchy strategy: chapter/section reset lower levels.
        if heading.startswith("第") and "章" in heading:
            return [heading]
        if heading.startswith("第") and "节" in heading:
            if heading_stack and "章" in heading_stack[0]:
                return [heading_stack[0], heading]
            return [heading]
        if re.match(r"^[一二三四五六七八九十]+、", heading):
            return heading_stack[:2] + [heading]
        if re.match(r"^\([一二三四五六七八九十]+\)", heading):
            return heading_stack[:3] + [heading]
        if re.match(r"^[0-9]+[.、]", heading):
            return heading_stack[:4] + [heading]
        if heading.startswith("#"):
            level = len(heading) - len(heading.lstrip("#"))
            trimmed = heading.strip("# ")
            return heading_stack[: max(level - 1, 0)] + [trimmed]
        return heading_stack + [heading]

    def _parse_tables(self, tables: Any, base_meta: dict[str, Any]) -> list[Chunk]:
        chunks: list[Chunk] = []
        if not isinstance(tables, list):
            return chunks

        for t_idx, table in enumerate(tables, start=1):
            if not table:
                continue
            rows_text: list[str] = []

            if isinstance(table, list):
                headers = table[0] if table and isinstance(table[0], list) else []
                for r_idx, row in enumerate(table[1:] if headers else table, start=1):
                    if isinstance(row, dict):
                        row_pairs = [f"{k}={v}" for k, v in row.items()]
                    elif isinstance(row, list):
                        if headers and len(headers) == len(row):
                            row_pairs = [f"{headers[i]}={row[i]}" for i in range(len(row))]
                        else:
                            row_pairs = [str(cell) for cell in row]
                    else:
                        row_pairs = [str(row)]
                    rows_text.append(f"行{r_idx}: " + " | ".join(row_pairs))
            elif isinstance(table, dict):
                for r_idx, (k, v) in enumerate(table.items(), start=1):
                    rows_text.append(f"行{r_idx}: {k}={v}")
            else:
                rows_text.append(str(table))

            if not rows_text:
                continue

            meta = dict(base_meta)
            meta["section_path"] = f"表格>{t_idx}"
            meta["chunk_type"] = "table"
            chunk_id = f"{base_meta['doc_id']}::table::{t_idx}"
            chunks.append(Chunk(chunk_id=chunk_id, text="\n".join(rows_text), metadata=meta))

        return chunks


# -----------------------------
# D. Query analyzer
# -----------------------------


class QueryAnalyzer:
    def analyze(self, query: str) -> QueryFeatures:
        normalized = normalize_text(query)

        article_refs = [normalize_article_token(m.group(0)) for m in ARTICLE_PATTERN.finditer(normalized)]
        article_refs = list(dict.fromkeys(article_refs))

        date_tokens = list(dict.fromkeys(DATE_PATTERN.findall(normalized)))
        for dm in DATE_DASH_PATTERN.findall(normalized):
            y, m, d = dm
            date_tokens.append(f"{int(y):04d}{int(m):02d}{int(d):02d}")
        date_tokens = list(dict.fromkeys(date_tokens))

        version_tokens = list(dict.fromkeys(VERSION_PATTERN.findall(normalized)))
        file_hints = list(dict.fromkeys(FILE_HINT_PATTERN.findall(normalized)))

        base_terms = safe_tokenize(normalized)
        special_terms = article_refs + date_tokens + version_tokens + file_hints
        must_terms = list(dict.fromkeys([t for t in base_terms + special_terms if len(t) >= 2]))

        return QueryFeatures(
            raw_query=query,
            normalized_query=normalized,
            article_refs=article_refs,
            date_tokens=date_tokens,
            version_tokens=version_tokens,
            file_hints=file_hints,
            must_terms=must_terms,
        )


# -----------------------------
# Embedding (model + fallback)
# -----------------------------


class Vectorizer:
    def fit(self, texts: list[str]) -> None:
        raise NotImplementedError

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerVectorizer(Vectorizer):
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        return None

    def encode(self, texts: list[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


class SparseHashVectorizer(Vectorizer):
    """No-model fallback vectorizer using hashed bag-of-words tf-idf."""

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> None:
        df: Counter[str] = Counter()
        n = len(texts)
        for text in texts:
            terms = set(safe_tokenize(text))
            for t in terms:
                df[t] += 1
        self.idf = {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            terms = safe_tokenize(text)
            tf = Counter(terms)
            for t, c in tf.items():
                idx = hash(t) % self.dim
                mat[i, idx] += c * self.idf.get(t, 1.0)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        return mat


# -----------------------------
# B. Hybrid retriever (vector + BM25 + fusion)
# -----------------------------


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], vectorizer: Vectorizer, rrf_k: int = 60) -> None:
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.rrf_k = rrf_k

        self.corpus_texts = [self._build_retrieval_text(c) for c in chunks]
        self.tokenized_corpus = [safe_tokenize(t) for t in self.corpus_texts]
        self.vectorizer.fit(self.corpus_texts)
        self.chunk_vectors = self.vectorizer.encode(self.corpus_texts)

        self.bm25 = BM25Okapi(self.tokenized_corpus) if BM25Okapi else None

    def _build_retrieval_text(self, chunk: Chunk) -> str:
        meta = chunk.metadata
        section_path = str(meta.get("section_path", ""))
        title = str(meta.get("full_title", ""))
        file_name = str(meta.get("file_name", ""))
        edition = str(meta.get("edition_tag", ""))
        version_date = str(meta.get("version_date", ""))
        return f"{title} {file_name} {section_path} {edition} {version_date}\n{chunk.text}"

    def retrieve(self, query_features: QueryFeatures, top_n: int = 30) -> list[RetrievalResult]:
        q_text = query_features.normalized_query
        q_tokens = safe_tokenize(q_text)

        q_vec = self.vectorizer.encode([q_text])
        vector_scores = np.dot(self.chunk_vectors, q_vec[0]).astype(np.float32)

        if self.bm25 is not None:
            bm25_scores = np.asarray(self.bm25.get_scores(q_tokens), dtype=np.float32)
        else:
            bm25_scores = self._token_overlap_scores(q_tokens)

        vec_rank = np.argsort(-vector_scores)
        bm_rank = np.argsort(-bm25_scores)

        vec_rrf = np.zeros_like(vector_scores)
        bm_rrf = np.zeros_like(bm25_scores)
        for r, idx in enumerate(vec_rank, start=1):
            vec_rrf[idx] = 1.0 / (self.rrf_k + r)
        for r, idx in enumerate(bm_rank, start=1):
            bm_rrf[idx] = 1.0 / (self.rrf_k + r)

        fusion = 0.5 * vec_rrf + 0.5 * bm_rrf
        top_indices = np.argsort(-fusion)[:top_n]

        return [
            RetrievalResult(
                chunk=self.chunks[i],
                vector_score=float(vector_scores[i]),
                bm25_score=float(bm25_scores[i]),
                fusion_score=float(fusion[i]),
            )
            for i in top_indices
        ]

    def _token_overlap_scores(self, q_tokens: list[str]) -> np.ndarray:
        qset = set(q_tokens)
        scores = np.zeros(len(self.tokenized_corpus), dtype=np.float32)
        for i, doc_tokens in enumerate(self.tokenized_corpus):
            dset = set(doc_tokens)
            overlap = len(qset & dset)
            denom = len(qset) + 1e-6
            scores[i] = overlap / denom
        return scores


# -----------------------------
# C. Smart reranker
# -----------------------------


class SmartReranker:
    def __init__(
        self,
        article_bonus: float = 1.8,
        date_bonus: float = 1.6,
        version_bonus: float = 1.4,
        file_hint_bonus: float = 1.2,
        must_term_bonus: float = 0.15,
        mismatch_version_penalty: float = -1.6,
    ) -> None:
        self.article_bonus = article_bonus
        self.date_bonus = date_bonus
        self.version_bonus = version_bonus
        self.file_hint_bonus = file_hint_bonus
        self.must_term_bonus = must_term_bonus
        self.mismatch_version_penalty = mismatch_version_penalty

    def rerank(self, candidates: list[RetrievalResult], features: QueryFeatures, top_k: int = 5) -> list[RerankResult]:
        reranked: list[RerankResult] = []

        for item in candidates:
            chunk_text = normalize_text(item.chunk.text)
            meta_text = normalize_text(" ".join(str(v) for v in item.chunk.metadata.values()))
            full_text = f"{meta_text}\n{chunk_text}"

            exact_bonus = 0.0
            exact_bonus += self._bonus_by_terms(full_text, features.article_refs, self.article_bonus)
            exact_bonus += self._bonus_by_terms(full_text, features.date_tokens, self.date_bonus)
            exact_bonus += self._bonus_by_terms(full_text, features.version_tokens, self.version_bonus)
            exact_bonus += self._bonus_by_terms(full_text, features.file_hints, self.file_hint_bonus)

            must_hits = sum(1 for t in features.must_terms if contains_any(full_text, [t]))
            exact_bonus += must_hits * self.must_term_bonus

            version_penalty = self._version_penalty(item.chunk.metadata, features)

            final_score = 0.45 * item.vector_score + 0.35 * item.bm25_score + 2.2 * item.fusion_score
            final_score += exact_bonus + version_penalty

            reranked.append(
                RerankResult(
                    chunk=item.chunk,
                    vector_score=item.vector_score,
                    bm25_score=item.bm25_score,
                    exact_bonus=exact_bonus,
                    version_penalty=version_penalty,
                    final_score=final_score,
                )
            )

        reranked.sort(key=lambda x: x.final_score, reverse=True)
        return reranked[:top_k]

    def _bonus_by_terms(self, haystack: str, terms: list[str], unit_bonus: float) -> float:
        if not terms:
            return 0.0
        bonus = 0.0
        for t in terms:
            if contains_any(haystack, [t]):
                bonus += unit_bonus
        return bonus

    def _version_penalty(self, metadata: dict[str, Any], features: QueryFeatures) -> float:
        if not features.version_tokens and not features.date_tokens:
            return 0.0

        meta_version = normalize_text(str(metadata.get("edition_tag", "")))
        meta_date = normalize_text(str(metadata.get("version_date", "")))

        wants_current = any(v in features.normalized_query for v in ["现行版", "最新版"])
        if wants_current and meta_version and ("现行" not in meta_version and "最新" not in meta_version):
            return self.mismatch_version_penalty

        requested_dates = set(features.date_tokens)
        if requested_dates and meta_date and meta_date not in requested_dates:
            return self.mismatch_version_penalty

        requested_versions = set(normalize_text(v) for v in features.version_tokens)
        if requested_versions and meta_version:
            if not any(v in meta_version for v in requested_versions):
                return self.mismatch_version_penalty

        return 0.0


# -----------------------------
# Pipeline + required interfaces
# -----------------------------


class LegalRetrievalPipeline:
    def __init__(self, retriever: HybridRetriever, analyzer: QueryAnalyzer, reranker: SmartReranker) -> None:
        self.retriever = retriever
        self.analyzer = analyzer
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        features = self.analyzer.analyze(query)
        candidates = self.retriever.retrieve(features, top_n=max(20, top_k * 6))
        reranked = self.reranker.rerank(candidates, features, top_k=top_k)

        return [
            {
                "chunk_id": item.chunk.chunk_id,
                "text": item.chunk.text,
                "metadata": item.chunk.metadata,
                "score_breakdown": {
                    "vector_score": item.vector_score,
                    "bm25_score": item.bm25_score,
                    "exact_bonus": item.exact_bonus,
                    "version_penalty": item.version_penalty,
                    "final_score": item.final_score,
                },
            }
            for item in reranked
        ]


_GLOBAL_PIPELINE: Optional[LegalRetrievalPipeline] = None


def build_index(
    documents: list[dict[str, Any]],
    embedding_model_name: Optional[str] = "BAAI/bge-small-zh-v1.5",
    use_sentence_transformer: bool = True,
) -> LegalRetrievalPipeline:
    """
    Build parser/index/retrieval pipeline from raw documents.

    Each document example:
    {
      "doc_id": "law_001",
      "file_name": "劳动合同法_20230901_现行版.pdf",
      "full_title": "中华人民共和国劳动合同法",
      "source_path": "/data/laws/law_001.pdf",
      "text": "...",
      "tables": [[ ["列1","列2"], ["值1","值2"] ]]
    }
    """
    global _GLOBAL_PIPELINE

    parser = DocumentParser()
    chunks = parser.parse_documents(documents)

    vectorizer: Vectorizer
    if use_sentence_transformer and SentenceTransformer is not None and embedding_model_name:
        try:
            vectorizer = SentenceTransformerVectorizer(embedding_model_name)
        except Exception:
            vectorizer = SparseHashVectorizer()
    else:
        vectorizer = SparseHashVectorizer()

    retriever = HybridRetriever(chunks=chunks, vectorizer=vectorizer)
    analyzer = QueryAnalyzer()
    reranker = SmartReranker()

    _GLOBAL_PIPELINE = LegalRetrievalPipeline(retriever=retriever, analyzer=analyzer, reranker=reranker)
    return _GLOBAL_PIPELINE


def retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve top-k evidence chunks from the global index."""
    if _GLOBAL_PIPELINE is None:
        raise RuntimeError("Index is not built. Please call build_index(...) first.")
    return _GLOBAL_PIPELINE.retrieve(query=query, top_k=top_k)
