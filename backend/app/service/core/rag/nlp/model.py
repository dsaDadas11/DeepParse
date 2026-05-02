from __future__ import annotations

import time
from typing import List

import numpy as np
from llama_index.core.data_structs import Node
from llama_index.core.schema import NodeWithScore
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from requests.exceptions import RequestException

from runtime_config import (
    CHAT_MODEL,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_REQUEST_MAX_RETRIES,
    EMBEDDING_REQUEST_RETRY_BASE_SECONDS,
    EMBEDDING_REQUEST_TIMEOUT_SECONDS,
    GENERATION_API_KEY,
    GENERATION_BASE_URL,
    RERANK_API_KEY,
    RERANK_REQUEST_MAX_RETRIES,
    RERANK_REQUEST_RETRY_BASE_SECONDS,
)
from utils import logger


class EmbeddingGenerationError(RuntimeError):
    """Raised when one or more embedding requests fail."""

    def __init__(self, message: str, failed_indices: list[int] | None = None):
        super().__init__(message)
        self.failed_indices = list(failed_indices or [])


def _generation_client() -> OpenAI:
    return OpenAI(api_key=GENERATION_API_KEY, base_url=GENERATION_BASE_URL)


def _embedding_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    return OpenAI(
        api_key=api_key or EMBEDDING_API_KEY,
        base_url=base_url or EMBEDDING_BASE_URL,
        timeout=EMBEDDING_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
    )


def _request_embeddings(
    client: OpenAI,
    batch: List[str],
    *,
    model_name: str,
    dimensions: int,
    encoding_format: str,
) -> list[list[float]]:
    completion = client.embeddings.create(
        model=model_name,
        input=batch,
        dimensions=dimensions,
        encoding_format=encoding_format,
    )
    batch_embeddings = [item.embedding for item in completion.data]
    if len(batch_embeddings) != len(batch):
        raise EmbeddingGenerationError(
            f"Embedding generation returned {len(batch_embeddings)} vectors for {len(batch)} inputs."
        )
    return batch_embeddings


def _request_embeddings_with_retry(
    client: OpenAI,
    batch: List[str],
    *,
    model_name: str,
    dimensions: int,
    encoding_format: str,
) -> list[list[float]]:
    retry_count = max(0, EMBEDDING_REQUEST_MAX_RETRIES)
    total_attempts = retry_count + 1

    for attempt in range(1, total_attempts + 1):
        try:
            return _request_embeddings(
                client,
                batch,
                model_name=model_name,
                dimensions=dimensions,
                encoding_format=encoding_format,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
            if attempt >= total_attempts:
                raise

            sleep_seconds = EMBEDDING_REQUEST_RETRY_BASE_SECONDS * attempt
            logger.warning(
                "Embedding request failed with %s on attempt %s/%s; retrying in %ss",
                exc.__class__.__name__,
                attempt,
                total_attempts,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def _generate_embeddings_resilient(
    client: OpenAI,
    batch: List[str],
    *,
    start_index: int,
    model_name: str,
    dimensions: int,
    encoding_format: str,
) -> tuple[list[list[float] | None], list[int]]:
    try:
        return (
            _request_embeddings_with_retry(
                client,
                batch,
                model_name=model_name,
                dimensions=dimensions,
                encoding_format=encoding_format,
            ),
            [],
        )
    except Exception as exc:
        if len(batch) == 1:
            logger.exception(
                "generate_embedding failed for item offset=%s",
                start_index,
            )
            return [None], [start_index]

        logger.warning(
            "generate_embedding batch failed for offset=%s size=%s; retrying with smaller batches",
            start_index,
            len(batch),
        )
        midpoint = max(1, len(batch) // 2)
        left_embeddings, left_failures = _generate_embeddings_resilient(
            client,
            batch[:midpoint],
            start_index=start_index,
            model_name=model_name,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
        right_embeddings, right_failures = _generate_embeddings_resilient(
            client,
            batch[midpoint:],
            start_index=start_index + midpoint,
            model_name=model_name,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )
        if left_failures or right_failures:
            logger.debug(
                "generate_embedding fallback summary offset=%s size=%s failed=%s",
                start_index,
                len(batch),
                left_failures + right_failures,
            )
        return left_embeddings + right_embeddings, left_failures + right_failures


def get_chat_completion_block(session_id, question, references):
    del session_id
    formatted_references = "\n".join(
        [f"[{ref['id']}] {ref['content']}" for ref in references]
    )
    prompt = f"""
You are a document QA assistant. Answer the question using the references first.
If the references are insufficient, say so explicitly.

References:
{formatted_references}

Question:
{question}
""".strip()

    completion = _generation_client().chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    return completion.choices[0].message.content or ""


def rerank_similarity(query, texts):
    nodes = [NodeWithScore(node=Node(text=text), score=1.0) for text in texts]
    if not nodes:
        return np.array([]), None

    retry_count = max(0, RERANK_REQUEST_MAX_RETRIES)
    total_attempts = retry_count + 1

    for attempt in range(1, total_attempts + 1):
        try:
            dashscope_rerank = DashScopeRerank(top_n=len(texts), api_key=RERANK_API_KEY)
            results = dashscope_rerank.postprocess_nodes(nodes, query_str=query)
            scores = np.array([res.score for res in results])
            return scores, None
        except RequestException as exc:
            if attempt >= total_attempts:
                raise

            sleep_seconds = RERANK_REQUEST_RETRY_BASE_SECONDS * attempt
            logger.warning(
                "Rerank request failed with %s on attempt %s/%s; retrying in %ss",
                exc.__class__.__name__,
                attempt,
                total_attempts,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError("Rerank request failed after retries.")


def generate_embedding(
    text: str | List[str],
    api_key: str | None = None,
    base_url: str | None = None,
    model_name: str = EMBEDDING_MODEL,
    dimensions: int = EMBEDDING_DIMENSIONS,
    encoding_format: str = "float",
    max_batch_size: int = 10,
):
    client = _embedding_client(api_key=api_key, base_url=base_url)

    if isinstance(text, str):
        return _request_embeddings_with_retry(
            client,
            [text],
            model_name=model_name,
            dimensions=dimensions,
            encoding_format=encoding_format,
        )[0]

    if isinstance(text, list):
        if not text:
            return []
        all_embeddings = []
        failed_indices: list[int] = []
        for i in range(0, len(text), max_batch_size):
            batch = text[i : i + max_batch_size]
            batch_embeddings, batch_failures = _generate_embeddings_resilient(
                client,
                batch,
                start_index=i,
                model_name=model_name,
                dimensions=dimensions,
                encoding_format=encoding_format,
            )
            failed_indices.extend(batch_failures)
            all_embeddings.extend(batch_embeddings)
        if failed_indices:
            sample = ", ".join(str(index) for index in failed_indices[:8])
            raise EmbeddingGenerationError(
                f"Embedding generation failed for {len(failed_indices)} input(s); sample indices: {sample}.",
                failed_indices=failed_indices,
            )
        return all_embeddings

    raise TypeError("text must be a string or list of strings")
