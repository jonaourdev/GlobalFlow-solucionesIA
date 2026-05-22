from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.agents import (
    BusinessRulesAgent,
    ClassificationAgent,
    ExtractionAgent,
    NormalizationAgent,
    ToolRAGAgent,
    ValidationAgent,
)
from app.config import settings
from app.memory import AgentMemory
from app.planner import GlobalFlowPlanner


class GlobalFlowOrchestrator:

    def __init__(self) -> None:
        self.planner = GlobalFlowPlanner()
        self.memory = AgentMemory()
        self.extractor = ExtractionAgent()
        self.normalizer = NormalizationAgent()
        self.rag_agent = ToolRAGAgent()
        self.classifier = ClassificationAgent()
        self.validator = ValidationAgent()
        self.rules = BusinessRulesAgent()

    def run(self, raw_invoice_text: str, file_name: str = "entrada_manual") -> dict[str, Any]:
        plan = self.planner.create_plan()
        self.memory.add("planificacion", "GlobalFlowPlanner", f"Plan creado con {len(plan)} pasos.")

        extracted = self.extractor.run(raw_invoice_text)
        self.memory.add("extraccion", self.extractor.name, f"Producto detectado: {extracted.producto}")

        normalized = self.normalizer.run(extracted)
        self.memory.add("normalizacion", self.normalizer.name, normalized.descripcion_normalizada)

        evidence = self.rag_agent.run(normalized)
        self.memory.add("herramientas_rag", self.rag_agent.name, f"Evidencias recuperadas: {len(evidence)}")

        classification = self.classifier.run(extracted, normalized, evidence)
        self.memory.add(
            "clasificacion",
            self.classifier.name,
            f"Código sugerido: {classification.codigo_sugerido} | confianza: {classification.nivel_confianza:.2f}",
        )

        validation = self.validator.run(normalized, classification, evidence)
        self.memory.add("validacion", self.validator.name, f"Veredicto: {validation.veredicto}")

        final_result = self.rules.run(
            classification=classification,
            validation=validation,
            evidence=evidence,
            resumen_flujo=self.memory.summary(),
        )
        self.memory.add("reglas_negocio", self.rules.name, f"Estado final: {final_result.estado}")

        payload = {
            "archivo": file_name,
            "plan": [step.model_dump() for step in plan],
            "extracted": extracted.model_dump(),
            "normalized": normalized.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            "classification": classification.model_dump(),
            "validation": validation.model_dump(),
            "final_result": final_result.model_dump(),
            "trace": [trace.model_dump() for trace in self.memory.traces],
        }

        self.save_result(payload, file_name)
        return payload

    def save_result(self, payload: dict[str, Any], file_name: str) -> None:
        results_dir = Path(settings.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        file_stem = Path(file_name).stem or "entrada_manual"
        output_path = results_dir / f"{file_stem}_resultado.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
