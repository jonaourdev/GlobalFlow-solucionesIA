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
from app.observability import ObservabilityTracker
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
        tracker = ObservabilityTracker(file_name=file_name, raw_invoice_text=raw_invoice_text)

        plan = tracker.measure("planificacion", "GlobalFlowPlanner", self.planner.create_plan)
        self.memory.add(
            "planificacion",
            "GlobalFlowPlanner",
            f"Plan creado con {len(plan)} pasos.",
            trace=tracker.trace_for("planificacion", "GlobalFlowPlanner", f"Plan creado con {len(plan)} pasos."),
        )

        extracted = tracker.measure("extraccion", self.extractor.name, lambda: self.extractor.run(raw_invoice_text))
        self.memory.add(
            "extraccion",
            self.extractor.name,
            f"Producto detectado: {extracted.producto}",
            trace=tracker.trace_for(
                "extraccion",
                self.extractor.name,
                f"Producto detectado: {extracted.producto}",
                metadata={"campos_faltantes": extracted.campos_faltantes},
            ),
        )

        normalized = tracker.measure("normalizacion", self.normalizer.name, lambda: self.normalizer.run(extracted))
        self.memory.add(
            "normalizacion",
            self.normalizer.name,
            normalized.descripcion_normalizada,
            trace=tracker.trace_for(
                "normalizacion",
                self.normalizer.name,
                normalized.descripcion_normalizada,
                metadata={"ambiguedades": normalized.posibles_ambiguedades},
            ),
        )

        evidence = tracker.measure("herramientas_rag", self.rag_agent.name, lambda: self.rag_agent.run(normalized))
        self.memory.add(
            "herramientas_rag",
            self.rag_agent.name,
            f"Evidencias recuperadas: {len(evidence)}",
            trace=tracker.trace_for(
                "herramientas_rag",
                self.rag_agent.name,
                f"Evidencias recuperadas: {len(evidence)}",
                metadata={"fuentes": sorted({item.fuente for item in evidence})},
            ),
        )

        classification = tracker.measure(
            "clasificacion",
            self.classifier.name,
            lambda: self.classifier.run(extracted, normalized, evidence),
        )
        self.memory.add(
            "clasificacion",
            self.classifier.name,
            f"Código sugerido: {classification.codigo_sugerido} | confianza: {classification.nivel_confianza:.2f}",
            trace=tracker.trace_for(
                "clasificacion",
                self.classifier.name,
                f"Código sugerido: {classification.codigo_sugerido} | confianza: {classification.nivel_confianza:.2f}",
                metadata={
                    "codigo_sugerido": classification.codigo_sugerido,
                    "confianza": classification.nivel_confianza,
                    "requiere_revision_humana": classification.requiere_revision_humana,
                },
            ),
        )

        validation = tracker.measure(
            "validacion",
            self.validator.name,
            lambda: self.validator.run(normalized, classification, evidence),
        )
        self.memory.add(
            "validacion",
            self.validator.name,
            f"Veredicto: {validation.veredicto}",
            trace=tracker.trace_for(
                "validacion",
                self.validator.name,
                f"Veredicto: {validation.veredicto}",
                metadata={"codigo_final": validation.codigo_final, "veredicto": validation.veredicto},
            ),
        )

        final_result = tracker.measure(
            "reglas_negocio",
            self.rules.name,
            lambda: self.rules.run(
                classification=classification,
                validation=validation,
                evidence=evidence,
                resumen_flujo=self.memory.summary(),
            ),
        )
        self.memory.add(
            "reglas_negocio",
            self.rules.name,
            f"Estado final: {final_result.estado}",
            trace=tracker.trace_for(
                "reglas_negocio",
                self.rules.name,
                f"Estado final: {final_result.estado}",
                metadata={
                    "estado": final_result.estado,
                    "codigo_final": final_result.codigo_final,
                    "requiere_revision_humana": final_result.requiere_revision_humana,
                },
            ),
        )

        metrics = tracker.build_metrics(
            extracted=extracted,
            normalized=normalized,
            evidence=evidence,
            classification=classification,
            validation=validation,
            final_result=final_result,
        )
        metrics_path = tracker.save_metrics(metrics)

        payload = {
            "archivo": file_name,
            "run_id": tracker.run_id,
            "plan": [step.model_dump() for step in plan],
            "extracted": extracted.model_dump(),
            "normalized": normalized.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            "classification": classification.model_dump(),
            "validation": validation.model_dump(),
            "final_result": final_result.model_dump(),
            "trace": [trace.model_dump() for trace in self.memory.traces],
            "observability": metrics.model_dump(),
            "observability_metrics_path": str(metrics_path),
        }

        self.save_result(payload, file_name)
        return payload

    def save_result(self, payload: dict[str, Any], file_name: str) -> None:
        results_dir = Path(settings.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        file_stem = Path(file_name).stem or "entrada_manual"
        output_path = results_dir / f"{file_stem}_resultado.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
