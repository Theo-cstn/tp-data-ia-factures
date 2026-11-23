"""Configuration centralisée du projet (chargée depuis .env)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR: Path = Path(__file__).resolve().parent.parent
ENV_FILE: Path = BASE_DIR / ".env"

# Charge les variables d'environnement depuis .env à la racine du projet
load_dotenv(dotenv_path=ENV_FILE)

# Chemins des données
DATA_DIR: Path = BASE_DIR / "data"
RAW_PDFS_DIR: Path = DATA_DIR / "raw_pdfs"
MANIFEST_STEP1_PATH: Path = RAW_PDFS_DIR / "manifest_step1.json"
BRONZE_DIR: Path = DATA_DIR / "bronze"
BRONZE_PARQUET_PATH: Path = BRONZE_DIR / "documents.parquet"
BRONZE_ERRORS_LOG_PATH: Path = BRONZE_DIR / "ingestion_errors.jsonl"
SILVER_DIR: Path = DATA_DIR / "silver"
SILVER_INVOICES_PATH: Path = SILVER_DIR / "factures.parquet"
SILVER_LINES_PATH: Path = SILVER_DIR / "lignes_facture.parquet"
SILVER_QUARANTINE_PATH: Path = SILVER_DIR / "quarantaine.parquet"
GROUND_TRUTH_PATH: Path = DATA_DIR / "ground_truth.json"
EVAL_OCR_CACHE_PATH: Path = DATA_DIR / "eval_ocr_cache.json"
EVAL_OCR_RESULTS_PATH: Path = DATA_DIR / "eval_ocr_results.json"
WAREHOUSE_DB_PATH: Path = DATA_DIR / "warehouse.duckdb"

# Configuration S3 (Hyperfluid)
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
AWS_ENDPOINT_URL: str = os.getenv(
    "AWS_ENDPOINT_URL",
    "https://dams-5.s3-gateway.console.hyperfluid.cloud",
).strip()
AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "default").strip()
S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "factures").strip()

# Configuration OCR (Hyperfluid / LightOnOCR-3)
HYPERFLUID_ORG_ID: str = "ebc6b7a8-9ecd-4d3d-bafc-fef40ebe986f"
HYPERFLUID_API_KEY: str = os.getenv("HYPERFLUID_API_KEY", "").strip()
OCR_MODEL: str = os.getenv("OCR_MODEL", "lightonocr-3-1b-test").strip()
OCR_ENDPOINT_URL: str = os.getenv(
    "OCR_ENDPOINT_URL",
    f"https://llm.hyperfluid.cloud/v1/organizations/{HYPERFLUID_ORG_ID}/shared-models/lightonocr-3-1b-test/v1/chat/completions",
).strip()
OCR_TARGET_LONGEST_EDGE: int = int(os.getenv("OCR_TARGET_LONGEST_EDGE", "1540").strip() or "1540")

# Configuration Agent LLM (Étape 8)
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "hyperfluid").strip().lower()
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma-4-12b").strip()
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()

# Seuil de caractères significatifs (hors espaces) pour distinguer un PDF texte d'un scan
TEXT_CHAR_THRESHOLD: int = 50

_PLACEHOLDER_VALUES: set[str] = {
    "",
    "your_access_key_id",
    "your_secret_access_key",
    "your_hyperfluid_api_key",
}


def has_valid_s3_credentials() -> bool:
    """Vérifie que les identifiants S3 sont renseignés et ne sont pas les valeurs par défaut."""
    return (
        AWS_ACCESS_KEY_ID not in _PLACEHOLDER_VALUES
        and AWS_SECRET_ACCESS_KEY not in _PLACEHOLDER_VALUES
        and bool(AWS_ENDPOINT_URL)
        and bool(S3_BUCKET_NAME)
    )


def has_valid_ocr_credentials() -> bool:
    """Vérifie que la clé API Hyperfluid et la configuration OCR sont renseignées."""
    return (
        HYPERFLUID_API_KEY not in _PLACEHOLDER_VALUES
        and bool(OCR_ENDPOINT_URL)
        and bool(OCR_MODEL)
    )


def resolve_llm_config(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, str]:
    """Résout la configuration LLM effective selon le fournisseur choisi."""
    eff_provider = (provider or LLM_PROVIDER or "hyperfluid").strip().lower()
    eff_model = (model or LLM_MODEL or "gemma-4-12b").strip()
    eff_base_url = (base_url if base_url is not None else LLM_BASE_URL).strip()
    eff_api_key = (api_key if api_key is not None else LLM_API_KEY).strip()

    if eff_provider == "hyperfluid":
        if not eff_base_url:
            eff_base_url = (
                f"https://llm.hyperfluid.cloud/v1/organizations/{HYPERFLUID_ORG_ID}"
                f"/shared-models/{eff_model}/v1/chat/completions"
            )
        if not eff_api_key:
            eff_api_key = HYPERFLUID_API_KEY
    elif eff_provider == "openai_compatible":
        if eff_base_url and not eff_base_url.rstrip("/").endswith("/chat/completions"):
            eff_base_url = eff_base_url.rstrip("/") + "/chat/completions"
        if not eff_api_key:
            eff_api_key = os.getenv("OPENAI_API_KEY", "").strip() or "local"
    elif eff_provider == "claude":
        if not eff_base_url:
            eff_base_url = "https://api.anthropic.com/v1/messages"
        if not eff_api_key:
            eff_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    elif eff_provider == "google":
        if not eff_base_url:
            eff_base_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/{eff_model}:generateContent"
            )
        if not eff_api_key:
            eff_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    else:
        raise ValueError(
            f"Fournisseur LLM inconnu : '{eff_provider}'. "
            "Valeurs supportées : 'hyperfluid', 'openai_compatible', 'claude', 'google'."
        )

    return {
        "provider": eff_provider,
        "model": eff_model,
        "base_url": eff_base_url,
        "api_key": eff_api_key,
    }


def has_valid_llm_credentials(
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Vérifie que la configuration LLM est prête à l'emploi."""
    cfg = resolve_llm_config(provider=provider, model=model, base_url=base_url, api_key=api_key)
    if cfg["provider"] == "openai_compatible":
        return bool(cfg["base_url"]) and bool(cfg["model"])
    return cfg["api_key"] not in _PLACEHOLDER_VALUES and bool(cfg["base_url"]) and bool(cfg["model"])
