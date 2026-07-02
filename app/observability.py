from __future__ import annotations

import json
import re
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from app.config import project_root, settings
from app.models import (
    ClassificationResult,
    EvidenceItem,
    ExecutionTrace,
    ExtractedInvoice,
    FinalResult,
    NormalizedProduct,
    ObservabilityMetrics,
    SecuritySignal,
    ValidationResult,
)


T = TypeVar("T")

TOKEN_PATTERN = re.compile(r"(gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,})")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
RUT_PATTERN = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")
PROMPT_INJECTION_PATTERN = re.compile(
    r"(ignora\s+(las\s+)?instrucciones|revela\s+(el\s+)?prompt|system\s+prompt|api[_\s-]?key|token)",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def observability_dir() -> Path:
    path = project_root() / settings.observability_dir
    path.mkdir(parents=True, exist_ok=True)
    (path / "runs").mkdir(parents=True, exist_ok=True)
    return path


def redact_sensitive_data(text: str) -> str:
    redacted = TOKEN_PATTERN.sub("[TOKEN_REDACTADO]", text)
    redacted = EMAIL_PATTERN.sub("[EMAIL_REDACTADO]", redacted)
    redacted = PHONE_PATTERN.sub(redact_phone_match, redacted)
    redacted = RUT_PATTERN.sub("[RUT_REDACTADO]", redacted)
    return redacted


def redact_phone_match(match: re.Match[str]) -> str:
    value = match.group(0)
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 9:
        return "[TELEFONO_REDACTADO]"
    return value


def has_phone_like_value(text: str) -> bool:
    for match in PHONE_PATTERN.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) >= 9:
            return True
    return False


def summarize_value(value: Any, max_chars: int = 700) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    elif isinstance(value, list):
        value = [item.model_dump() if hasattr(item, "model_dump") else item for item in value]

    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)

    text = redact_sensitive_data(text)
    if len(text) > max_chars:
        return f"{text[:max_chars]}..."
    return text


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_ground_truth(file_name: str) -> Optional[dict[str, Any]]:
    ground_truth_path = observability_dir() / "ground_truth.json"
    if not ground_truth_path.exists():
        return None

    data = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    file_stem = Path(file_name).stem
    return data.get(file_name) or data.get(file_stem)


def scan_security(raw_invoice_text: str, extracted: Optional[ExtractedInvoice] = None) -> list[SecuritySignal]:
    signals: list[SecuritySignal] = []

    if PROMPT_INJECTION_PATTERN.search(raw_invoice_text):
        signals.append(
            SecuritySignal(
                tipo="prompt_injection",
                severidad="alta",
                descripcion="El texto de entrada contiene instrucciones que podrían intentar modificar el comportamiento del agente.",
                recomendacion="Mantener reglas de sistema estrictas, no ejecutar instrucciones del documento y escalar a revisión humana.",
            )
        )

    if EMAIL_PATTERN.search(raw_invoice_text) or has_phone_like_value(raw_invoice_text) or RUT_PATTERN.search(raw_invoice_text):
        signals.append(
            SecuritySignal(
                tipo="datos_personales",
                severidad="media",
                descripcion="Se detectaron posibles datos personales o identificadores sensibles en la factura.",
                recomendacion="Registrar solo versiones redactadas en logs y aplicar minimización de datos.",
            )
        )

    if TOKEN_PATTERN.search(raw_invoice_text):
        signals.append(
            SecuritySignal(
                tipo="secreto_en_entrada",
                severidad="alta",
                descripcion="La entrada parece contener un token o secreto técnico.",
                recomendacion="Bloquear persistencia del secreto, rotar credenciales afectadas y revisar origen del documento.",
            )
        )

    if extracted and extracted.campos_faltantes:
        signals.append(
            SecuritySignal(
                tipo="calidad_datos",
                severidad="baja",
                descripcion=f"Faltan campos relevantes para la decisión: {', '.join(extracted.campos_faltantes)}.",
                recomendacion="Solicitar datos faltantes antes de aprobar automáticamente.",
            )
        )

    return signals


class ObservabilityTracker:
    def __init__(self, file_name: str, raw_invoice_text: str) -> None:
        self.run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.file_name = file_name
        self.raw_invoice_text = raw_invoice_text
        self.started_at = utc_now()
        self.started_perf = time.perf_counter()
        self.step_events: list[dict[str, Any]] = []
        self.logs_path = observability_dir() / "execution_logs.jsonl"
        self.metrics_path = observability_dir() / "metrics.jsonl"

    def measure(self, paso: str, agente: str, operation: Callable[[], T], metadata: Optional[dict[str, Any]] = None) -> T:
        started = time.perf_counter()
        timestamp = utc_now()
        base_event = {
            "run_id": self.run_id,
            "archivo": self.file_name,
            "paso": paso,
            "agente": agente,
            "timestamp": timestamp,
            "metadata": metadata or {},
        }

        try:
            result = operation()
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            event = {
                **base_event,
                "estado": "error",
                "duracion_ms": duration_ms,
                "error_tipo": exc.__class__.__name__,
                "error": redact_sensitive_data(str(exc)),
            }
            self.step_events.append(event)
            append_jsonl(self.logs_path, event)
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        event = {
            **base_event,
            "estado": "ok",
            "duracion_ms": duration_ms,
            "salida_resumen": summarize_value(result),
        }
        self.step_events.append(event)
        append_jsonl(self.logs_path, event)
        return result

    def trace_for(self, paso: str, agente: str, detalle: str, metadata: Optional[dict[str, Any]] = None) -> ExecutionTrace:
        event = next(
            (item for item in reversed(self.step_events) if item["paso"] == paso and item["agente"] == agente),
            None,
        )
        return ExecutionTrace(
            paso=paso,
            agente=agente,
            detalle=detalle,
            estado=event.get("estado", "ok") if event else "ok",
            timestamp=event.get("timestamp") if event else utc_now(),
            duracion_ms=event.get("duracion_ms") if event else None,
            metadata=metadata or {},
        )

    def build_metrics(
        self,
        extracted: ExtractedInvoice,
        normalized: NormalizedProduct,
        evidence: list[EvidenceItem],
        classification: ClassificationResult,
        validation: ValidationResult,
        final_result: FinalResult,
    ) -> ObservabilityMetrics:
        finished_at = utc_now()
        duration_total_ms = round((time.perf_counter() - self.started_perf) * 1000, 2)
        evidence_by_type = dict(Counter(item.tipo for item in evidence))
        latencies = {event["paso"]: event["duracion_ms"] for event in self.step_events}
        security_signals = scan_security(self.raw_invoice_text, extracted)
        fallas = detect_failures(extracted, normalized, evidence, classification, validation, final_result, self.step_events)
        consistencia = compute_consistency_score(classification, validation, final_result, evidence)
        precision_real = compute_real_precision(self.file_name, final_result)
        precision_estimada = compute_estimated_precision(
            classification=classification,
            validation=validation,
            final_result=final_result,
            evidence=evidence,
            extracted=extracted,
            normalized=normalized,
            consistency_score=consistencia,
        )
        recomendaciones = build_recommendations(
            latencies=latencies,
            evidence=evidence,
            extracted=extracted,
            normalized=normalized,
            classification=classification,
            validation=validation,
            final_result=final_result,
            failures=fallas,
            security_signals=security_signals,
            consistency_score=consistencia,
        )

        return ObservabilityMetrics(
            run_id=self.run_id,
            archivo=self.file_name,
            timestamp_inicio=self.started_at,
            timestamp_fin=finished_at,
            duracion_total_ms=duration_total_ms,
            latencia_por_paso_ms=latencies,
            precision_real=precision_real,
            precision_estimada=precision_estimada,
            consistencia_score=consistencia,
            confianza_clasificacion=round(classification.nivel_confianza, 4),
            evidencia_total=len(evidence),
            evidencia_por_tipo=evidence_by_type,
            campos_faltantes_total=len(extracted.campos_faltantes),
            ambiguedades_total=len(normalized.posibles_ambiguedades),
            estado_final=final_result.estado,
            requiere_revision_humana=final_result.requiere_revision_humana,
            fallas_detectadas=fallas,
            security_signals=security_signals,
            recomendaciones=recomendaciones,
        )

    def save_metrics(self, metrics: ObservabilityMetrics) -> Path:
        payload = metrics.model_dump()
        run_path = observability_dir() / "runs" / f"{metrics.run_id}.json"
        run_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        append_jsonl(self.metrics_path, payload)
        return run_path


def compute_real_precision(file_name: str, final_result: FinalResult) -> Optional[float]:
    expected = load_ground_truth(file_name)
    if not expected:
        return None

    expected_code = expected.get("codigo_final")
    expected_state = expected.get("estado")
    code_ok = True if expected_code is None else final_result.codigo_final == expected_code
    state_ok = True if expected_state is None else final_result.estado == expected_state
    return 1.0 if code_ok and state_ok else 0.0


def compute_estimated_precision(
    classification: ClassificationResult,
    validation: ValidationResult,
    final_result: FinalResult,
    evidence: list[EvidenceItem],
    extracted: ExtractedInvoice,
    normalized: NormalizedProduct,
    consistency_score: float,
) -> float:
    score = classification.nivel_confianza * 0.45 + consistency_score * 0.35
    if evidence:
        score += 0.10
    if validation.veredicto == "confirmado":
        score += 0.10
    if final_result.estado == "revision_humana" and (
        classification.requiere_revision_humana or validation.veredicto == "revision_humana"
    ):
        score += 0.05
    score -= min(len(extracted.campos_faltantes) * 0.03, 0.12)
    score -= min(len(normalized.posibles_ambiguedades) * 0.02, 0.10)
    return round(max(0.0, min(score, 1.0)), 4)


def compute_consistency_score(
    classification: ClassificationResult,
    validation: ValidationResult,
    final_result: FinalResult,
    evidence: list[EvidenceItem],
) -> float:
    score = 1.0
    validation_code = validation.codigo_final or classification.codigo_sugerido

    if validation.veredicto == "confirmado" and validation_code != classification.codigo_sugerido:
        score -= 0.25
    if final_result.estado == "aprobado" and not final_result.codigo_final:
        score -= 0.40
    if final_result.estado == "aprobado" and final_result.codigo_final != validation_code:
        score -= 0.25
    if final_result.estado == "aprobado" and classification.requiere_revision_humana:
        score -= 0.25
    if final_result.estado == "aprobado" and not evidence:
        score -= 0.30
    if final_result.estado == "revision_humana" and validation.veredicto == "confirmado" and classification.nivel_confianza >= settings.confidence_auto_approve:
        score -= 0.10

    return round(max(0.0, min(score, 1.0)), 4)


def detect_failures(
    extracted: ExtractedInvoice,
    normalized: NormalizedProduct,
    evidence: list[EvidenceItem],
    classification: ClassificationResult,
    validation: ValidationResult,
    final_result: FinalResult,
    step_events: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []

    if any(event["estado"] == "error" for event in step_events):
        failures.append("Existe al menos un paso con error de ejecución.")
    if extracted.campos_faltantes:
        failures.append("La extracción dejó campos faltantes que reducen la confiabilidad.")
    if normalized.posibles_ambiguedades:
        failures.append("La normalización detectó ambigüedades semánticas.")
    if not evidence:
        failures.append("No se recuperó evidencia desde herramientas/RAG.")
    if classification.nivel_confianza < settings.confidence_human_review:
        failures.append("La confianza del clasificador está bajo el umbral mínimo.")
    if validation.veredicto == "revision_humana":
        failures.append("El validador normativo escaló el caso a revisión humana.")
    if final_result.estado == "aprobado" and final_result.requiere_revision_humana:
        failures.append("Inconsistencia: resultado aprobado marcado simultáneamente como revisión humana.")

    return failures


def build_recommendations(
    latencies: dict[str, float],
    evidence: list[EvidenceItem],
    extracted: ExtractedInvoice,
    normalized: NormalizedProduct,
    classification: ClassificationResult,
    validation: ValidationResult,
    final_result: FinalResult,
    failures: list[str],
    security_signals: list[SecuritySignal],
    consistency_score: float,
) -> list[str]:
    recommendations: list[str] = []

    if latencies:
        slow_step, slow_latency = max(latencies.items(), key=lambda item: item[1])
        if slow_latency > 10000:
            recommendations.append(
                f"Optimizar latencia en '{slow_step}': concentró {slow_latency:.0f} ms. Evaluar cache, reducción de contexto o modelo más liviano."
            )

    if not evidence:
        recommendations.append("Fortalecer RAG/herramientas: sin evidencia no debe existir aprobación automática.")
    elif len(evidence) < 3:
        recommendations.append("Aumentar diversidad de evidencia recuperada para mejorar robustez ante variabilidad de datos.")

    if extracted.campos_faltantes:
        recommendations.append("Agregar una etapa de solicitud de datos faltantes antes de clasificar facturas incompletas.")

    if normalized.posibles_ambiguedades:
        recommendations.append("Crear reglas de desambiguación por familia de producto para reducir escalamiento manual.")

    if classification.nivel_confianza < settings.confidence_auto_approve:
        recommendations.append("Mantener revisión humana para casos bajo umbral y usar esos casos como set de mejora supervisada.")

    if validation.veredicto == "corregido":
        recommendations.append("Analizar correcciones del validador para ajustar prompt del clasificador y ejemplos de few-shot.")

    if final_result.estado == "revision_humana":
        recommendations.append("Registrar motivo de revisión humana como etiqueta para mejorar precisión real en futuras evaluaciones.")

    if consistency_score < 0.85:
        recommendations.append("Revisar consistencia entre clasificador, validador y reglas de negocio antes de producción.")

    if security_signals:
        recommendations.append("Aplicar minimización de datos y redacción obligatoria en logs antes de operar con datos reales.")

    if failures:
        recommendations.append("Priorizar los puntos de falla detectados en trazabilidad antes de escalar el flujo a más documentos.")

    if not recommendations:
        recommendations.append("Mantener monitoreo continuo y comparar métricas por lote para detectar degradación del agente.")

    return recommendations
