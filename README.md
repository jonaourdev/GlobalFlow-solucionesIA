# GlobalFlow RAG

Proyecto en Python para clasificar productos de exportación usando **LLM + RAG** con LangChain.

## Descripción breve

Este proyecto toma una descripción de producto desde una factura y genera una sugerencia de **código arancelario** usando:

- una **base arancelaria** en Excel,
- un **manual normativo** en `.docx`,
- y **facturas históricas** como apoyo contextual.

El flujo general es el siguiente:

1. Se carga la base arancelaria.
2. Se carga el manual normativo y se divide en fragmentos (_chunks_).
3. Se generan embeddings y se crea una **base vectorial** con FAISS.
4. Cuando se ingresa una descripción de producto, el sistema:
   - busca códigos candidatos en la base arancelaria,
   - recupera contexto relevante desde la base vectorial,
   - y envía esa información al modelo LLM para generar una clasificación final.

## Estructura esperada del proyecto

```bash
text/
├── documentation/
│   ├── base_arancelaria_sintetica_globalflow.xlsx
│   ├── facturas_historicas_sinteticas_globalflow.xlsx
│   └── manual_normativo_sintetico_globalflow.docx
└── globalflow/
    ├── globalflow_code.py
    ├── requirements.txt
    └── README.md
```

## Configuración del proyecto

### 1) Abrir el proyecto

Abrir el proyecto y ubicarse en la carpeta

```bash
cd ruta\de\tu\proyecto\text\globalflow
```

### 2) Instalar dependencias

Con el comando dentro de la terminal:

```bash
pip install -r requirements.txt
```

### 3) Crear archivo .env

Dentro del archivo colocar los tokens y url necesarios.

### 4) Ejecutar con el proyecto

Con el comando en la terminal:

```bash
python globalflow_code.py
```
