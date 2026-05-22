from __future__ import annotations

import re
from typing import Optional

import pandas as pd
from app.langchain_compat import tool

from app.models import EvidenceItem
from app.repository_data import get_tariff_dataframe, get_historical_dataframe


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-záéíóúñü0-9]+", str(text).lower()))


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    columns = set(df.columns)
    for candidate in candidates:
        if candidate in columns:
            return candidate
    for candidate in candidates:
        for column in df.columns:
            if candidate in column:
                return column
    return None


def _safe_value(row: pd.Series, column: Optional[str], default: str = "") -> str:
    if not column or column not in row or pd.isna(row[column]):
        return default
    return str(row[column])


def _score_dataframe_rows(df: pd.DataFrame, query: str) -> list[tuple[int, pd.Series]]:
    query_tokens = _tokenize(query)
    scored_rows: list[tuple[int, pd.Series]] = []

    for _, row in df.iterrows():
        searchable = " ".join(str(row.get(col, "")) for col in df.columns)
        row_tokens = _tokenize(searchable)
        score = len(query_tokens.intersection(row_tokens))
        if score > 0:
            scored_rows.append((score, row))

    scored_rows.sort(key=lambda item: item[0], reverse=True)
    return scored_rows


def search_tariff_database(query: str, top_k: int = 5) -> list[EvidenceItem]:
    df = get_tariff_dataframe()
    if df.empty:
        return []

    codigo_col = _pick_column(df, ["codigo", "codigo_arancelario", "codigo_sugerido", "partida", "subpartida"])
    categoria_col = _pick_column(df, ["categoria", "familia", "tipo_producto", "seccion", "capitulo"])
    descripcion_col = _pick_column(df, ["descripcion", "descripcion_producto", "producto", "detalle", "nombre"])
    keywords_col = _pick_column(df, ["palabras_clave", "keywords", "terminos", "atributos", "observacion"])

    evidence: list[EvidenceItem] = []
    for score, row in _score_dataframe_rows(df, query)[:top_k]:
        codigo = _safe_value(row, codigo_col, "sin_codigo")
        categoria = _safe_value(row, categoria_col, "sin_categoria")
        descripcion = _safe_value(row, descripcion_col, "")
        palabras = _safe_value(row, keywords_col, "")

        all_fields = "\n".join(
            f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])
        )

        evidence.append(
            EvidenceItem(
                tipo="base_arancelaria",
                fuente=f"base_arancelaria:{codigo}",
                contenido=(
                    f"Código: {codigo}\n"
                    f"Categoría: {categoria}\n"
                    f"Descripción: {descripcion}\n"
                    f"Palabras clave/observación: {palabras}\n"
                    f"Coincidencias léxicas: {score}\n\n"
                    f"Fila completa:\n{all_fields}"
                ),
                metadata={"codigo": codigo, "categoria": categoria, "score": score},
            )
        )
    return evidence


def search_historical_cases(query: str, top_k: int = 3) -> list[EvidenceItem]:
    """Herramienta externa: busca facturas históricas similares."""
    df = get_historical_dataframe()
    if df.empty:
        return []

    evidence: list[EvidenceItem] = []
    for score, row in _score_dataframe_rows(df, query)[:top_k]:
        all_fields = "\n".join(
            f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])
        )
        evidence.append(
            EvidenceItem(
                tipo="historico",
                fuente=f"facturas_historicas:score_{score}",
                contenido=f"Caso histórico similar. Coincidencias léxicas: {score}\n{all_fields}",
                metadata={"score": score},
            )
        )
    return evidence


@tool
def buscar_base_arancelaria(query: str) -> str:
    """Busca códigos candidatos en la base arancelaria estructurada usando texto del producto."""
    results = search_tariff_database(query)
    if not results:
        return "No se encontraron candidatos en la base arancelaria."
    return "\n\n".join(item.contenido for item in results)


@tool
def buscar_facturas_historicas(query: str) -> str:
    """Busca casos históricos de facturas similares al producto consultado."""
    results = search_historical_cases(query)
    if not results:
        return "No se encontraron casos históricos similares."
    return "\n\n".join(item.contenido for item in results)
