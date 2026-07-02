import argparse
import json
from pathlib import Path

from app.dashboard import generate_dashboard
from app.orchestrator import GlobalFlowOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta el flujo de agentes GlobalFlow usando LangChain Classic + herramientas + RAG."
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
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="No regenerar el dashboard HTML al finalizar la ejecución.",
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

    orchestrator = GlobalFlowOrchestrator()
    result = orchestrator.run(raw_invoice_text=raw_text, file_name=file_name)

    print(json.dumps(result["final_result"], ensure_ascii=False, indent=2))
    if not args.no_dashboard:
        dashboard_path = generate_dashboard()
        print(f"\nDashboard de observabilidad: {dashboard_path}")


if __name__ == "__main__":
    main()
