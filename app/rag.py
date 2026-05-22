from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import resolve_data_file, settings
from app.models import EvidenceItem
from app.repository_data import load_repository_documents


def get_embeddings():
    # Modelo multilingüe local. Evita depender de un endpoint externo para embeddings.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def load_legacy_documents() -> list[Document]:
    """Soporte adicional para PDFs/TXT antiguos dentro de data/manuales."""
    docs: list[Document] = []
    manuales_dir = resolve_data_file("manuales")

    if manuales_dir.exists():
        pdf_docs = PyPDFDirectoryLoader(str(manuales_dir)).load()
        for d in pdf_docs:
            d.metadata["tipo"] = "manual_normativo"
        docs.extend(pdf_docs)

        for path in manuales_dir.glob("*.txt"):
            text_docs = TextLoader(str(path), encoding="utf-8").load()
            for d in text_docs:
                d.metadata["tipo"] = "manual_normativo"
            docs.extend(text_docs)

    return docs


def load_documents() -> list[Document]:
    """
    Carga documentos desde:
    1) documentation/ del repo GlobalFlow-solucionesIA (.docx y .xlsx)
    2) data/manuales/ como compatibilidad hacia atrás (.pdf y .txt)
    """
    docs = load_repository_documents()
    docs.extend(load_legacy_documents())
    return docs


def build_vectorstore() -> Chroma:
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma(
        collection_name="globalflow_rag",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )

    if chunks:
        vectorstore.add_documents(chunks)
    return vectorstore


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="globalflow_rag",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def retrieve_rag_evidence(query: str, k: int = 5) -> list[EvidenceItem]:
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    evidence: list[EvidenceItem] = []
    for idx, doc in enumerate(docs, start=1):
        tipo = doc.metadata.get("tipo", "manual_normativo")
        if tipo not in {"manual_normativo", "historico", "base_arancelaria", "documentacion_caso"}:
            tipo = "manual_normativo"
        source = doc.metadata.get("source", f"rag_doc_{idx}")
        page = doc.metadata.get("page")
        row_index = doc.metadata.get("row_index")

        detalle = ""
        if page is not None:
            detalle = f" página {page}"
        elif row_index is not None:
            detalle = f" fila {row_index}"

        evidence.append(
            EvidenceItem(
                tipo=tipo,
                fuente=f"{source}{detalle}",
                contenido=doc.page_content[:1200],
                metadata=doc.metadata,
            )
        )
    return evidence
