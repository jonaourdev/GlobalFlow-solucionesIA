from __future__ import annotations

from app.models import PlanStep


class GlobalFlowPlanner:

    def create_plan(self) -> list[PlanStep]:
        return [
            PlanStep(
                orden=1,
                agente="AgenteExtractor",
                objetivo="Extraer campos relevantes desde la factura.",
                entrada="Texto crudo de la factura.",
                salida_esperada="Producto, descripción original, cantidad, material y países si existen.",
            ),
            PlanStep(
                orden=2,
                agente="AgenteNormalizador",
                objetivo="Normalizar la descripción comercial.",
                entrada="Datos extraídos de la factura.",
                salida_esperada="Descripción normalizada, atributos clave y términos de búsqueda.",
            ),
            PlanStep(
                orden=3,
                agente="AgenteHerramientasRAG",
                objetivo="Buscar evidencia en herramientas externas y RAG.",
                entrada="Descripción normalizada y términos de búsqueda.",
                salida_esperada="Candidatos de base arancelaria, casos históricos y fragmentos normativos.",
            ),
            PlanStep(
                orden=4,
                agente="AgenteClasificador",
                objetivo="Proponer el código arancelario más probable.",
                entrada="Descripción y evidencia recuperada.",
                salida_esperada="Código sugerido, justificación, confianza y fuentes.",
            ),
            PlanStep(
                orden=5,
                agente="AgenteValidador",
                objetivo="Auditar normativamente la clasificación propuesta.",
                entrada="Clasificación sugerida y evidencia.",
                salida_esperada="Veredicto confirmado, corregido o revisión humana.",
            ),
            PlanStep(
                orden=6,
                agente="AgenteReglasNegocio",
                objetivo="Aplicar umbrales y decidir aprobación o revisión humana.",
                entrada="Clasificación y validación.",
                salida_esperada="Resultado final trazable.",
            ),
        ]
