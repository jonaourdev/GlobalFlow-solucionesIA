import argparse
import json
from pathlib import Path

from app.graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el flujo agentic de GlobalFlow con LangChain + LangGraph."
    )
    parser.add_argument(
        "invoice_path",
        nargs="?",
        default=None,
        help="Ruta de archivo .txt con el texto de la factura. Opcional si usas --text.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Descripción o texto de factura ingresado directamente por consola.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.text:
        raw_text = args.text
        file_name = "entrada_manual"
    else:
        invoice_path = Path(args.invoice_path) if args.invoice_path else Path("data/facturas/factura_demo.txt")
        if not invoice_path.exists():
            raise FileNotFoundError(f"No existe el archivo: {invoice_path}")
        raw_text = invoice_path.read_text(encoding="utf-8")
        file_name = invoice_path.name

    graph = build_graph()
    result = graph.invoke(
        {
            "raw_invoice_text": raw_text,
            "file_name": file_name,
            "errors": [],
        }
    )

    final_result = result["final_result"].model_dump()
    print(json.dumps(final_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
