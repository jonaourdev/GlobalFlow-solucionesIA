
import os
import pandas as pd
from pathlib import Path
from docx import Document

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# 1) Configuración
BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com")
API_KEY = os.getenv("GITHUB_TOKEN")

if not API_KEY:
    raise ValueError("No se encontró GITHUB_TOKEN en las variables de entorno.")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model="gpt-4o-mini",
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url=BASE_URL,
    api_key=API_KEY
)


# 2) Cargar archivos
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR.parent / "documentation"

BASE_ARANCELARIA_PATH = DOCS_DIR / "base_arancelaria_sintetica_globalflow.xlsx"
FACTURAS_PATH = DOCS_DIR / "facturas_historicas_sinteticas_globalflow.xlsx"
MANUAL_PATH = DOCS_DIR / "manual_normativo_sintetico_globalflow.docx"

base_df = pd.read_excel(BASE_ARANCELARIA_PATH)
facturas_df = pd.read_excel(FACTURAS_PATH)

doc = Document(MANUAL_PATH)
manual_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# 3) Preparar documentos para RAG
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len
)

manual_chunks = splitter.split_text(manual_text)

historical_docs = []
for _, row in facturas_df.iterrows():
    texto = (
        f"Factura histórica. "
        f"Descripción: {row['descripcion_raw']}. "
        f"Descripción normalizada: {row['descripcion_normalizada']}. "
        f"Código final: {row['codigo_final']}. "
        f"Estado: {row['estado_clasificacion']}. "
        f"Nota revisor: {row['nota_revisor']}"
    )
    historical_docs.append(texto)

all_docs = manual_chunks + historical_docs

vector_db = FAISS.from_texts(texts=all_docs, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 4})


# 4) Buscar candidatos en base arancelaria
def buscar_candidatos_base(descripcion, base_df, top_n=5):
    descripcion_lower = descripcion.lower()
    resultados = []

    for _, row in base_df.iterrows():
        texto_base = " ".join([
            str(row.get("categoria", "")),
            str(row.get("descripcion_oficial", "")),
            str(row.get("palabras_clave", "")),
            str(row.get("atributos_requeridos", ""))
        ]).lower()

        score = 0
        for palabra in descripcion_lower.split():
            if palabra in texto_base:
                score += 1

        resultados.append({
            "codigo_hs": row["codigo_hs"],
            "categoria": row["categoria"],
            "descripcion_oficial": row["descripcion_oficial"],
            "score": score
        })

    resultados = sorted(resultados, key=lambda x: x["score"], reverse=True)
    return resultados[:top_n]


# 5) Clasificación simple con LLM + RAG
def clasificar_producto(descripcion_producto):
    candidatos = buscar_candidatos_base(descripcion_producto, base_df, top_n=5)
    docs_recuperados = retriever.invoke(descripcion_producto)

    contexto_rag = "\n\n".join([doc.page_content for doc in docs_recuperados])
    contexto_candidatos = "\n".join([
        f"- Código: {c['codigo_hs']} | Categoría: {c['categoria']} | Descripción: {c['descripcion_oficial']}"
        for c in candidatos
    ])

    prompt_sistema = (
        "Eres un asistente de clasificación arancelaria.\n"
        "Debes sugerir el código más probable usando SOLO los candidatos entregados y el contexto recuperado.\n"
        "No inventes códigos. Si la información es insuficiente, indica que requiere revisión humana.\n"
        "Responde en este formato:\n"
        "codigo_sugerido: ...\n"
        "categoria: ...\n"
        "justificacion: ...\n"
        "requiere_revision_humana: si/no"
    )

    prompt_usuario = f"""
Descripción del producto:
{descripcion_producto}

Candidatos desde la base arancelaria:
{contexto_candidatos}

Contexto recuperado con RAG:
{contexto_rag}
"""

    response = llm.invoke([
        SystemMessage(content=prompt_sistema),
        HumanMessage(content=prompt_usuario)
    ])

    return {
        "descripcion": descripcion_producto,
        "candidatos": candidatos,
        "documentos_recuperados": [doc.page_content for doc in docs_recuperados],
        "respuesta_final": response.content
    }


# 6) Ejemplo de uso
if __name__ == "__main__":
    descripcion = "Set de recipientes plásticos reutilizables para cocina"
    resultado = clasificar_producto(descripcion)

    print("=== DESCRIPCIÓN ===")
    print(resultado["descripcion"])

    print("\n=== CANDIDATOS DE BASE ARANCELARIA ===")
    for c in resultado["candidatos"]:
        print(c)

    print("\n=== DOCUMENTOS RECUPERADOS ===")
    for i, doc in enumerate(resultado["documentos_recuperados"], 1):
        print(f"\nDocumento {i}:\n{doc[:500]}")

    print("\n=== RESPUESTA FINAL DEL MODELO ===")
    print(resultado["respuesta_final"])
