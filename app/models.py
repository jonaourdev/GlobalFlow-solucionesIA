from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class PlanStep(BaseModel):

    orden: int
    agente: str
    objetivo: str
    entrada: str
    salida_esperada: str


class ExecutionTrace(BaseModel):

    paso: str
    agente: str
    detalle: str
    estado: Literal["ok", "error"] = "ok"
    timestamp: Optional[str] = None
    duracion_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SecuritySignal(BaseModel):
    tipo: str
    severidad: Literal["baja", "media", "alta"]
    descripcion: str
    recomendacion: str


class ObservabilityMetrics(BaseModel):
    run_id: str
    archivo: str
    timestamp_inicio: str
    timestamp_fin: str
    duracion_total_ms: float
    latencia_por_paso_ms: dict[str, float] = Field(default_factory=dict)
    precision_real: Optional[float] = None
    precision_estimada: float
    consistencia_score: float
    confianza_clasificacion: float
    evidencia_total: int
    evidencia_por_tipo: dict[str, int] = Field(default_factory=dict)
    campos_faltantes_total: int
    ambiguedades_total: int
    estado_final: str
    requiere_revision_humana: bool
    fallas_detectadas: list[str] = Field(default_factory=list)
    security_signals: list[SecuritySignal] = Field(default_factory=list)
    recomendaciones: list[str] = Field(default_factory=list)


class ExtractedInvoice(BaseModel):
    producto: str = Field(description="Nombre del producto detectado en la factura")
    descripcion_original: str = Field(description="Descripción textual original del producto")
    cantidad: Optional[str] = Field(default=None, description="Cantidad detectada, si existe")
    material: Optional[str] = Field(default=None, description="Material del producto, si existe")
    pais_origen: Optional[str] = Field(default=None, description="País de origen, si existe")
    pais_destino: Optional[str] = Field(default=None, description="País de destino, si existe")
    campos_faltantes: list[str] = Field(default_factory=list)


class NormalizedProduct(BaseModel):
    descripcion_normalizada: str
    atributos_clave: list[str] = Field(default_factory=list)
    terminos_busqueda: list[str] = Field(default_factory=list)
    posibles_ambiguedades: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    tipo: Literal["base_arancelaria", "manual_normativo", "historico", "documentacion_caso"]
    fuente: str
    contenido: str
    metadata: dict = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    producto: str
    codigo_sugerido: str
    categoria: str
    justificacion: str
    nivel_confianza: float = Field(ge=0.0, le=1.0)
    fuentes: list[str] = Field(default_factory=list)
    requiere_revision_humana: bool


class ValidationResult(BaseModel):
    veredicto: Literal["confirmado", "corregido", "revision_humana"]
    codigo_final: Optional[str] = None
    explicacion: str
    evidencia_utilizada: list[str] = Field(default_factory=list)


class FinalResult(BaseModel):
    estado: Literal["aprobado", "revision_humana"]
    codigo_final: Optional[str] = None
    producto: str
    nivel_confianza: float
    explicacion: str
    fuentes: list[str] = Field(default_factory=list)
    requiere_revision_humana: bool
    resumen_flujo: list[str] = Field(default_factory=list)
