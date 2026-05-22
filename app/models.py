from typing import Literal, Optional
from pydantic import BaseModel, Field

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
