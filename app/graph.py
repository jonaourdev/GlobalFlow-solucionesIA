from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.llm import get_llm
from app.models import (
    ExtractedInvoice,
    NormalizedProduct,
    EvidenceItem,
    ClassificationResult,
    ValidationResult,
    FinalResult,
)
from app.prompts import (
    EXTRACTION_PROMPT,
    NORMALIZATION_PROMPT,
    CLASSIFICATION_PROMPT,
    VALIDATION_PROMPT,
)
from app.rag import retrieve_rag_evidence
from app.tools import search_tariff_database


class GlobalFlowState(TypedDict, total=False):
    raw_invoice_text: str
    file_name: str
    extracted: ExtractedInvoice
    normalized: NormalizedProduct
    evidence: list[EvidenceItem]
    classification: ClassificationResult
    validation: ValidationResult
    final_result: FinalResult
    errors: list[str]


def _as_json(value) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), ensure_ascii=False, indent=2)
    if isinstance(value, list):
        serializable = [v.model_dump() if hasattr(v, "model_dump") else v for v in value]
        return json.dumps(serializable, ensure_ascii=False, indent=2)
    return json.dumps(value, ensure_ascii=False, indent=2)


def extraction_agent(state: GlobalFlowState) -> GlobalFlowState:
    llm = get_llm(settings.model_triage).with_structured_output(ExtractedInvoice)
    messages = EXTRACTION_PROMPT.format_messages(raw_invoice_text=state["raw_invoice_text"])
    extracted = llm.invoke(messages)
    return {"extracted": extracted}


def normalization_agent(state: GlobalFlowState) -> GlobalFlowState:
    llm = get_llm(settings.model_triage).with_structured_output(NormalizedProduct)
    messages = NORMALIZATION_PROMPT.format_messages(
        extracted_invoice=_as_json(state["extracted"])
    )
    normalized = llm.invoke(messages)
    return {"normalized": normalized}


def rag_agent(state: GlobalFlowState) -> GlobalFlowState:
    normalized = state["normalized"]
    query = " ".join([
        normalized.descripcion_normalizada,
        " ".join(normalized.atributos_clave),
        " ".join(normalized.terminos_busqueda),
    ])

    tariff_evidence = search_tariff_database(query, top_k=5)
    rag_evidence = retrieve_rag_evidence(query, k=5)

    evidence = tariff_evidence + rag_evidence
    return {"evidence": evidence}


def classification_agent(state: GlobalFlowState) -> GlobalFlowState:
    llm = get_llm(settings.model_triage).with_structured_output(ClassificationResult)
    messages = CLASSIFICATION_PROMPT.format_messages(
        descripcion_original=state["extracted"].descripcion_original,
        descripcion_normalizada=state["normalized"].descripcion_normalizada,
        evidencia=_as_json(state.get("evidence", [])),
    )
    classification = llm.invoke(messages)
    return {"classification": classification}


def validation_agent(state: GlobalFlowState) -> GlobalFlowState:
    llm = get_llm(settings.model_validator).with_structured_output(ValidationResult)
    messages = VALIDATION_PROMPT.format_messages(
        descripcion_normalizada=state["normalized"].descripcion_normalizada,
        classification=_as_json(state["classification"]),
        evidencia=_as_json(state.get("evidence", [])),
    )
    validation = llm.invoke(messages)
    return {"validation": validation}


def rules_agent(state: GlobalFlowState) -> GlobalFlowState:
    classification = state["classification"]
    validation = state["validation"]

    requiere_revision = False
    motivos: list[str] = []

    if classification.requiere_revision_humana:
        requiere_revision = True
        motivos.append("El agente clasificador solicitó revisión humana.")

    if classification.nivel_confianza < settings.confidence_human_review:
        requiere_revision = True
        motivos.append(
            f"Confianza bajo umbral mínimo: {classification.nivel_confianza:.2f}."
        )

    if validation.veredicto == "revision_humana":
        requiere_revision = True
        motivos.append("El validador normativo indicó revisión humana.")

    if not state.get("evidence"):
        requiere_revision = True
        motivos.append("No existe evidencia recuperada desde RAG/base arancelaria.")

    if validation.veredicto == "confirmado" and not requiere_revision:
        estado = "aprobado"
        codigo_final = validation.codigo_final or classification.codigo_sugerido
        explicacion = validation.explicacion
    elif validation.veredicto == "corregido" and classification.nivel_confianza >= settings.confidence_auto_approve:
        estado = "aprobado"
        codigo_final = validation.codigo_final
        explicacion = validation.explicacion
    else:
        estado = "revision_humana"
        codigo_final = validation.codigo_final or classification.codigo_sugerido
        explicacion = " ".join(motivos) or validation.explicacion

    final = FinalResult(
        estado=estado,
        codigo_final=codigo_final,
        producto=classification.producto,
        nivel_confianza=classification.nivel_confianza,
        explicacion=explicacion,
        fuentes=list(set(classification.fuentes + validation.evidencia_utilizada)),
        requiere_revision_humana=estado == "revision_humana",
    )
    return {"final_result": final}


def route_after_rules(state: GlobalFlowState) -> str:
    final = state["final_result"]
    if final.estado == "revision_humana":
        return "human_review"
    return "save_result"


def human_review_agent(state: GlobalFlowState) -> GlobalFlowState:
    # MVP: no bloquea esperando al humano. Solo deja el caso marcado y guardado.
    # En producción, aquí se podría crear una tarea en una bandeja interna,
    # enviar una notificación o usar interrupt() de LangGraph para pausar el flujo.
    return state


def save_result_agent(state: GlobalFlowState) -> GlobalFlowState:
    results_dir = Path(settings.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    file_stem = Path(state.get("file_name", "factura")).stem
    output_path = results_dir / f"{file_stem}_resultado.json"

    payload = {
        "archivo": state.get("file_name"),
        "extracted": state["extracted"].model_dump(),
        "normalized": state["normalized"].model_dump(),
        "evidence": [item.model_dump() for item in state.get("evidence", [])],
        "classification": state["classification"].model_dump(),
        "validation": state["validation"].model_dump(),
        "final_result": state["final_result"].model_dump(),
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def build_graph():
    builder = StateGraph(GlobalFlowState)

    builder.add_node("extract", extraction_agent)
    builder.add_node("normalize", normalization_agent)
    builder.add_node("rag", rag_agent)
    builder.add_node("classify", classification_agent)
    builder.add_node("validate", validation_agent)
    builder.add_node("rules", rules_agent)
    builder.add_node("human_review", human_review_agent)
    builder.add_node("save_result", save_result_agent)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "normalize")
    builder.add_edge("normalize", "rag")
    builder.add_edge("rag", "classify")
    builder.add_edge("classify", "validate")
    builder.add_edge("validate", "rules")

    builder.add_conditional_edges(
        "rules",
        route_after_rules,
        {
            "human_review": "human_review",
            "save_result": "save_result",
        },
    )

    builder.add_edge("human_review", "save_result")
    builder.add_edge("save_result", END)

    return builder.compile()
