# GlobalFlow Logistics - Agentes con LangChain + LangGraph

Proyecto base para implementar un flujo de clasificación arancelaria con agentes especializados:

1. Extracción de factura.
2. Normalización de descripción.
3. Recuperación RAG.
4. Clasificación arancelaria.
5. Validación normativa.
6. Reglas de negocio.
7. Revisión humana cuando corresponda.

## Documentación base

```text
GlobalFlow-solucionesIA/
├── documentation/
│   ├── Documentación Caso GlobalFlow Logistics.docx
│   ├── base_arancelaria_sintetica_globalflow.xlsx
│   ├── facturas_historicas_sinteticas_globalflow.xlsx
│   └── manual_normativo_sintetico_globalflow.docx
└── globalflow/
```

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac
```

Edita `.env` y agrega tu `GITHUB_TOKEN`.

## Archivos usados

La solución prioriza estos archivos:

- `documentation/base_arancelaria_sintetica_globalflow.xlsx`: base arancelaria estructurada.
- `documentation/facturas_historicas_sinteticas_globalflow.xlsx`: casos históricos para RAG.
- `documentation/manual_normativo_sintetico_globalflow.docx`: manual normativo para RAG.

## Crear índice RAG

Antes de ejecutar clasificaciones, crea o actualiza la base vectorial:

```bash
python -m app.ingest
```

Esto carga:

- el manual normativo `.docx`,
- la documentación del caso `.docx`,
- las filas del Excel de facturas históricas,
- y las filas del Excel de base arancelaria.

## Ejecutar flujo con una factura TXT

```bash
python -m app.main data/facturas/factura_demo.txt
```

## Ejecutar flujo con texto directo

```bash
python -m app.main --text "Factura N° 01. Producto: T-shirt algodón caballero blanco. Cantidad: 100 unidades. Origen: Chile. Destino: España."
```

## Resultado

El resultado final se imprime en consola y también se guarda en:

```text
data/resultados/
```

El JSON generado incluye:

- datos extraídos,
- descripción normalizada,
- evidencia recuperada,
- clasificación propuesta,
- validación normativa,
- resultado final aprobado o revisión humana.

## Flujo LangGraph

```text
START
  ↓
extract
  ↓
normalize
  ↓
rag
  ↓
classify
  ↓
validate
  ↓
rules
  ↓
¿requiere revisión humana?
  ├── sí → human_review → save_result
  └── no → save_result
```

## Notas importantes

- Si no existe la carpeta `documentation`, el código usará los archivos de ejemplo en `data/` como fallback.
- El sistema requiere conexión a GitHub Models mediante `GITHUB_TOKEN` para ejecutar los agentes LLM.
