import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_models_base_url: str = os.getenv(
        "GITHUB_MODELS_BASE_URL", "https://models.github.ai/inference"
    )
    model_triage: str = os.getenv("MODEL_TRIAGE", "openai/gpt-4o-mini")
    model_validator: str = os.getenv("MODEL_VALIDATOR", "openai/gpt-4o")

    data_dir: str = os.getenv("DATA_DIR", "data")
    documentation_dir: str = os.getenv("DOCUMENTATION_DIR", "documentation")

    tariff_excel_filename: str = os.getenv(
        "TARIFF_EXCEL_FILENAME", "base_arancelaria_sintetica_globalflow.xlsx"
    )
    historical_excel_filename: str = os.getenv(
        "HISTORICAL_EXCEL_FILENAME", "facturas_historicas_sinteticas_globalflow.xlsx"
    )
    normative_docx_filename: str = os.getenv(
        "NORMATIVE_DOCX_FILENAME", "manual_normativo_sintetico_globalflow.docx"
    )

    chroma_dir: str = os.getenv("CHROMA_DIR", ".chroma/globalflow")
    results_dir: str = os.getenv("RESULTS_DIR", "data/resultados")
    confidence_auto_approve: float = float(os.getenv("CONFIDENCE_AUTO_APPROVE", "0.85"))
    confidence_human_review: float = float(os.getenv("CONFIDENCE_HUMAN_REVIEW", "0.70"))


settings = Settings()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_documentation_dir() -> Path:
    """Detecta la carpeta documentation aunque ejecutes desde raíz o desde /globalflow."""
    configured = Path(settings.documentation_dir)
    root = project_root()
    cwd = Path.cwd()

    candidates = [
        configured,
        cwd / settings.documentation_dir,
        cwd.parent / settings.documentation_dir,
        root / settings.documentation_dir,
        root.parent / settings.documentation_dir,
        root.parent.parent / settings.documentation_dir,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    return configured.resolve()


def resolve_data_file(*parts: str) -> Path:
    return (project_root() / settings.data_dir / Path(*parts)).resolve()


def ensure_env() -> None:
    if not settings.github_token:
        raise RuntimeError(
            "Falta GITHUB_TOKEN. Crea un archivo .env basado en .env.example y agrega tu token."
        )
