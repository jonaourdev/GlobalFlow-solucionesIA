from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.llm import get_llm
from app.models import (
    ClassificationResult,
    EvidenceItem,
    ExtractedInvoice,
    FinalResult,
    NormalizedProduct,
    ValidationResult,
)
from app.prompts import (
    CLASSIFICATION_PROMPT,
    EXTRACTION_PROMPT,
    NORMALIZATION_PROMPT,
    VALIDATION_PROMPT,
)
from app.rag import retrieve_rag_evidence
from app.tools import search_historical_cases, search_tariff_database


def as_json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        return json.dumps(value.model_dump(), ensure_ascii=False, indent=2)
    if isinstance(value, list):
        serializable = [v.model_dump() if hasattr(v, "model_dump") else v for v in value]
        return json.dumps(serializable, ensure_ascii=False, indent=2)
    return json.dumps(value, ensure_ascii=False, indent=2)


@dataclass
class BaseGlobalFlowAgent:
    name: str
    role: str
    goal: str
    tools: list[str] = field(default_factory=list)

    def describe(self) -> str:
        tools = ", ".join(self.tools) if self.tools else "sin herramientas externas"
        return f"{self.name} | Rol: {self.role} | Objetivo: {self.goal} | Tools: {tools}"


class ExtractionAgent(BaseGlobalFlowAgent):
    def __init__(self) -> None:
        super().__init__(
            name="AgenteExtractor",
            role="Extracción documental",
            goal="Identificar campos útiles desde la factura sin clasificar todavía.",
        )

    def run(self, raw_invoice_text: str) -> ExtractedInvoice:
        llm = get_llm(settings.model_triage).with_structured_output(ExtractedInvoice)
        messages = EXTRACTION_PROMPT.format_messages(raw_invoice_text=raw_invoice_text)
        return llm.invoke(messages)


class NormalizationAgent(BaseGlobalFlowAgent):
    def __init__(self) -> None:
        super().__init__(
            name="AgenteNormalizador",
            role="Normalización semántica",
            goal="Convertir la descripción comercial en términos claros para búsqueda y clasificación.",
        )

    def run(self, extracted: ExtractedInvoice) -> NormalizedProduct:
        llm = get_llm(settings.model_triage).with_structured_output(NormalizedProduct)
        messages = NORMALIZATION_PROMPT.format_messages(extracted_invoice=as_json(extracted))
        return llm.invoke(messages)


class ToolRAGAgent(BaseGlobalFlowAgent):
    def __init__(self) -> None:
        super().__init__(
            name="AgenteHerramientasRAG",
            role="Búsqueda con herramientas externas",
            goal="Recuperar evidencia desde base arancelaria, históricos y manual normativo.",
            tools=["buscar_base_arancelaria", "buscar_facturas_historicas", "retriever_rag"],
        )

    def run(self, normalized: NormalizedProduct) -> list[EvidenceItem]:
        query = " ".join([
            normalized.descripcion_normalizada,
            " ".join(normalized.atributos_clave),
            " ".join(normalized.terminos_busqueda),
        ])

        tariff_evidence = search_tariff_database(query, top_k=5)
        historical_evidence = search_historical_cases(query, top_k=3)
        rag_evidence = retrieve_rag_evidence(query, k=5)

        return tariff_evidence + historical_evidence + rag_evidence


class ClassificationAgent(BaseGlobalFlowAgent):
    def __init__(self) -> None:
        super().__init__(
            name="AgenteClasificador",
            role="Razonamiento y clasificación",
            goal="Proponer un código arancelario solo si existe evidencia suficiente.",
            tools=["evidencia_recuperada"],
        )

    def run(
        self,
        extracted: ExtractedInvoice,
        normalized: NormalizedProduct,
        evidence: list[EvidenceItem],
    ) -> ClassificationResult:
        llm = get_llm(settings.model_triage).with_structured_output(ClassificationResult)
        messages = CLASSIFICATION_PROMPT.format_messages(
            descripcion_original=extracted.descripcion_original,
            descripcion_normalizada=normalized.descripcion_normalizada,
            evidencia=as_json(evidence),
        )
        return llm.invoke(messages)


class ValidationAgent(BaseGlobalFlowAgent):
    def __init__(self) -> None:
        super().__init__(
            name="AgenteValidador",
            role="Auditoría normativa",
            goal="Confirmar, corregir o escalar la clasificación propuesta.",
            tools=["evidencia_recuperada"],
        )

    def run(
        self,
        normalized: NormalizedProduct,
        classification: ClassificationResult,
        evidence: list[EvidenceItem],
    ) -> ValidationResult:
        llm = get_llm(settings.model_validator).with_structured_output(ValidationResult)
        messages = VALIDATION_PROMPT.format_messages(
            descripcion_normalizada=normalized.descripcion_normalizada,
            classification=as_json(classification),
            evidencia=as_json(evidence),
        )
        return llm.invoke(messages)


class BusinessRulesAgent(BaseGlobalFlowAgent):
    def __init__(self) -> None:
        super().__init__(
            name="AgenteReglasNegocio",
            role="Control determinístico",
            goal="Aplicar umbrales y evitar aprobaciones sin respaldo.",
        )

    def run(
        self,
        classification: ClassificationResult,
        validation: ValidationResult,
        evidence: list[EvidenceItem],
        resumen_flujo: list[str],
    ) -> FinalResult:
        requiere_revision = False
        motivos: list[str] = []

        if classification.requiere_revision_humana:
            requiere_revision = True
            motivos.append("El agente clasificador solicitó revisión humana.")

        if classification.nivel_confianza < settings.confidence_human_review:
            requiere_revision = True
            motivos.append(f"Confianza bajo umbral mínimo: {classification.nivel_confianza:.2f}.")

        if validation.veredicto == "revision_humana":
            requiere_revision = True
            motivos.append("El validador normativo indicó revisión humana.")

        if not evidence:
            requiere_revision = True
            motivos.append("No existe evidencia recuperada desde herramientas/RAG.")

        if validation.veredicto == "confirmado" and not requiere_revision:
            estado = "aprobado"
            codigo_final = validation.codigo_final or classification.codigo_sugerido
            explicacion = validation.explicacion
        elif (
            validation.veredicto == "corregido"
            and validation.codigo_final
            and classification.nivel_confianza >= settings.confidence_auto_approval
            and not requiere_revision
        ):
            estado = "aprobado"
            codigo_final = validation.codigo_final
            explicacion = validation.explicacion
        else:
            estado = "revision_humana"
            codigo_final = None
            explicacion = " ".join(motivos) or validation.explicacion

        return FinalResult(
            estado=estado,
            codigo_final=codigo_final,
            producto=classification.producto,
            nivel_confianza=classification.nivel_confianza,
            explicacion=explicacion,
            fuentes=sorted(set(classification.fuentes + validation.evidencia_utilizada)),
            requiere_revision_humana=estado == "revision_humana",
            resumen_flujo=resumen_flujo,
        )
