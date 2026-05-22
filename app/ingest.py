from app.rag import build_vectorstore, load_documents
from app.config import resolve_documentation_dir


def main() -> None:
    docs = load_documents()
    print(f"Carpeta documentation detectada: {resolve_documentation_dir()}")
    print(f"Documentos/filas cargadas antes de chunking: {len(docs)}")

    vectorstore = build_vectorstore()
    print("Índice RAG creado/actualizado correctamente.")
    print(f"Colección: {vectorstore._collection.name}")


if __name__ == "__main__":
    main()
