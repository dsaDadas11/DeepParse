import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_DIR = BACKEND_DIR.parent


def _load_runtime_env() -> None:
    env_files = (
        BACKEND_DIR / ".env",
        PROJECT_DIR / ".env",
        PROJECT_DIR / "key.txt",
        Path("/app/key.txt"),
    )
    for env_file in env_files:
        if env_file.exists():
            load_dotenv(env_file, override=True)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer value for {name}: {value}") from exc


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    raw = os.getenv(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"Invalid float value for {name}: {value}") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean value for {name}: {value}")


def _normalize_root_path(value: str | None) -> str:
    if not value:
        return ""

    candidate = value.strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    if parsed.scheme and parsed.netloc:
        return parsed.path or ""

    if not candidate.startswith("/"):
        return f"/{candidate}"
    return candidate


_load_runtime_env()

DEMO_USER_ID = os.getenv("DEMO_USER_ID", "demo_user")

GENERATION_API_KEY = os.getenv("GENERATION_API_KEY") or os.getenv("OPENAI_API_KEY")
GENERATION_BASE_URL = os.getenv("GENERATION_BASE_URL") or os.getenv("OPENAI_BASE_URL")
CHAT_MODEL = os.getenv("GENERATION_MODEL") or os.getenv("CHAT_MODEL") or "kimi-k2.5"
RECOMMENDATION_MODEL = os.getenv("RECOMMENDATION_MODEL") or CHAT_MODEL
SESSION_NAME_MODEL = os.getenv("SESSION_NAME_MODEL") or CHAT_MODEL
EVAL_GENERATION_MODEL = os.getenv("EVAL_GENERATION_MODEL") or CHAT_MODEL

EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("DASHSCOPE_BASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = _int_env("EMBEDDING_DIMENSIONS", 1024)
EMBEDDING_REQUEST_TIMEOUT_SECONDS = _int_env("EMBEDDING_REQUEST_TIMEOUT_SECONDS", 90)
EMBEDDING_REQUEST_MAX_RETRIES = _int_env("EMBEDDING_REQUEST_MAX_RETRIES", 3)
EMBEDDING_REQUEST_RETRY_BASE_SECONDS = _int_env("EMBEDDING_REQUEST_RETRY_BASE_SECONDS", 2)

RERANK_API_KEY = os.getenv("RERANK_API_KEY") or EMBEDDING_API_KEY
RERANK_REQUEST_MAX_RETRIES = _int_env("RERANK_REQUEST_MAX_RETRIES", 2)
RERANK_REQUEST_RETRY_BASE_SECONDS = _int_env("RERANK_REQUEST_RETRY_BASE_SECONDS", 2)
REWRITE_API_KEY = os.getenv("REWRITE_API_KEY") or EMBEDDING_API_KEY or GENERATION_API_KEY
REWRITE_BASE_URL = os.getenv("REWRITE_BASE_URL") or EMBEDDING_BASE_URL or GENERATION_BASE_URL
REWRITE_MODEL = os.getenv("REWRITE_MODEL") or "qwen-flash"

LEGAL_TERM_NORMALIZATION_ENABLED = _bool_env("LEGAL_TERM_NORMALIZATION_ENABLED", True)
LEGAL_TERM_DICT_PATH = os.getenv("LEGAL_TERM_DICT_PATH", "")
INTENT_DEADLINE_WEIGHT_BIAS = _float_env("INTENT_DEADLINE_WEIGHT_BIAS", 0.06)
INTENT_AMOUNT_WEIGHT_BIAS = _float_env("INTENT_AMOUNT_WEIGHT_BIAS", 0.05)
RETRIEVAL_ROUTE_MODE = os.getenv("RETRIEVAL_ROUTE_MODE", "auto").strip().lower()
ENABLE_RETRIEVAL_TRACE = _bool_env("ENABLE_RETRIEVAL_TRACE", True)
ENABLE_FALLBACK_ROUTE = _bool_env("ENABLE_FALLBACK_ROUTE", True)
ENABLE_LEGAL_METADATA_ROUTE = _bool_env("ENABLE_LEGAL_METADATA_ROUTE", True)
ENABLE_LEGAL_METADATA_SCORING = _bool_env("ENABLE_LEGAL_METADATA_SCORING", True)
ENABLE_LEGAL_METADATA_HARD_FILTER = _bool_env("ENABLE_LEGAL_METADATA_HARD_FILTER", True)
STRICT_CITATION_BINDING = _bool_env("STRICT_CITATION_BINDING", False)
CONFLICT_ABSTAIN_ENABLED = _bool_env("CONFLICT_ABSTAIN_ENABLED", False)
EVAL_WRITE_CONFIG_SNAPSHOT = _bool_env("EVAL_WRITE_CONFIG_SNAPSHOT", True)
EVAL_ERROR_TAXONOMY_ENABLED = _bool_env("EVAL_ERROR_TAXONOMY_ENABLED", True)

DATABASE_URL = os.getenv("DATABASE_URL")
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
ES_PASSWORD = os.getenv("ELASTIC_PASSWORD") or os.getenv("ES_PASSWORD")

APP_ROOT_PATH = _normalize_root_path(os.getenv("APP_ROOT_PATH") or os.getenv("ROOT_PATH"))
CORS_ORIGINS = _csv_env(
    "CORS_ORIGINS",
    "http://localhost:5181,http://127.0.0.1:5181,http://localhost:5173,http://127.0.0.1:5173",
)
MAX_UPLOAD_SIZE_BYTES = _int_env("MAX_UPLOAD_SIZE_BYTES", 5 * 1024 * 1024)
ALLOWED_UPLOAD_EXTENSIONS = tuple(
    ext.lower()
    for ext in _csv_env(
        "ALLOWED_UPLOAD_EXTENSIONS",
        ".pdf,.docx,.txt,.md,.html,.json,.xlsx,.xls,.pptx,.ppt",
    )
)

REQUIRED_RUNTIME_VALUES = {
    "GENERATION_API_KEY": GENERATION_API_KEY,
    "GENERATION_BASE_URL": GENERATION_BASE_URL,
    "EMBEDDING_API_KEY": EMBEDDING_API_KEY,
    "EMBEDDING_BASE_URL": EMBEDDING_BASE_URL,
    "DATABASE_URL": DATABASE_URL,
    "ES_HOST": ES_HOST,
}


def validate_runtime_config() -> None:
    missing = [name for name, value in REQUIRED_RUNTIME_VALUES.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime config: {joined}. Please update backend/.env or key.txt before starting the service."
        )
