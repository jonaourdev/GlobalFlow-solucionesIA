# GlobalFlow Logistics - Agentes con LangChain Classic + RAG

## Estructura principal

```text
app/
├── agents.py              # Agentes especializados del flujo
├── orchestrator.py        # Orquestador secuencial de agentes
├── planner.py             # Planificador del flujo
├── memory.py              # Memoria/trazabilidad de ejecución
├── langchain_compat.py    # Imports compatibles con langchain_classic
├── rag.py                 # Índice RAG con Chroma
├── tools.py               # Herramientas externas: aranceles e históricos
├── repository_data.py     # Lectura de documentation/*.xlsx y *.docx
├── prompts.py             # Prompts de cada agente
├── models.py              # Modelos Pydantic de entrada/salida
├── ingest.py              # Crea el índice RAG
├── observability.py       # Cálculo de metricas, logs, seguridad y recomendaciones
├── dashboard.py           # Generación del Dashboard
├── evaluate.py            # Evaluación por lote de facturas
├── orchestrator.py        # Instrumentación del flujo de agentes
└── main.py                # Punto de ejecución
```

## Datos esperados

El código está preparado para leer la carpeta `documentation` del repositorio `GlobalFlow-solucionesIA`:

```text
documentation/
├── Documentación Caso GlobalFlow Logistics.docx
├── base_arancelaria_sintetica_globalflow.xlsx
├── facturas_historicas_sinteticas_globalflow.xlsx
└── manual_normativo_sintetico_globalflow.docx
```

También mantiene fallback a:

```text
data/aranceles.csv
data/historicos.csv
data/manuales/*.pdf
data/manuales/*.txt
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Edita `.env` y agrega tu `GITHUB_TOKEN`.

## Crear índice RAG

```bash
python -m app.ingest
```

Ejecuta este comando cuando agregues o modifiques documentos en `documentation`.

## Ejecutar con texto directo

```bash
python -m app.main --text "Factura N° 01. Producto: T-shirt algodón caballero blanco. Cantidad: 100 unidades. Origen: Chile. Destino: España."
```

## Ejecutar con archivo TXT

```bash
python -m app.main data/facturas/factura_demo.txt
```

## Ejecutar con archivos TXT con distintos resultados

```bash
python -m app.main data/facturas/factura_01_aprobacion_automatica.txt
```

```bash
python -m app.main data/facturas/factura_02_revision_humana_ambigua.txt
```

```bash
python -m app.main data/facturas/factura_03_correccion_o_escalamiento.txt
```

## Observabilidad, trazabilidad y dashboard

Cada ejecución registra métricas y trazas estructuradas para evaluar precisión, latencia, consistencia, puntos de falla y señales de seguridad.

Archivos generados:

```text
data/observability/
├── execution_logs.jsonl      # Logs por paso/agente con duración, estado y salida resumida
├── metrics.jsonl             # Métricas agregadas por ejecución
├── dashboard.html            # Dashboard local en HTML
└── runs/*.json               # Métricas completas por run_id
```

Genera solo el dashboard desde métricas ya existentes:

```bash
python -m app.dashboard
```

Ejecuta una factura y actualiza el dashboard automáticamente:

```bash
python -m app.main data/facturas/factura_demo.txt
```

Evalúa todas las facturas de prueba para medir variabilidad de datos:

```bash
python -m app.evaluate --input-dir data/facturas
```

Para medir precisión real, crea `data/observability/ground_truth.json` usando como base `data/observability/ground_truth.example.json` y completa `codigo_final`/`estado` esperados por archivo. Si no existe ground truth, el sistema reporta `precision_estimada`, calculada desde confianza, evidencia, validación y consistencia.

Métricas incluidas:

```text
- Precisión real: compara salida con ground_truth.json cuando existe.
- Precisión estimada: proxy basado en confianza, evidencia, validación y consistencia.
- Latencia: duración total y por paso del flujo.
- Consistencia: alineación entre clasificador, validador y reglas de negocio.
- Trazabilidad: logs JSONL por agente, estado, duración y salida resumida redactada.
- Seguridad: detección de prompt injection, posibles datos personales y secretos en entrada.
- Optimización: recomendaciones automáticas por latencia, baja evidencia, ambigüedad o fallas.
```

## Flujo implementado

```text
Planificación
  ↓
AgenteExtractor
  ↓
AgenteNormalizador
  ↓
AgenteHerramientasRAG
  ↓
AgenteClasificador
  ↓
AgenteValidador
  ↓
AgenteReglasNegocio
  ↓
Resultado aprobado o revisión humana
```
