from langchain_openai import ChatOpenAI
from app.config import settings, ensure_env


def get_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    ensure_env()
    return ChatOpenAI(
        model=model_name,
        api_key=settings.github_token,
        base_url=settings.github_models_base_url,
        temperature=temperature,
    )
