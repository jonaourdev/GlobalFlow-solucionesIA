from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.dashboard import generate_dashboard
from app.orchestrator import GlobalFlowOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta evaluación por lote para medir observabilidad en facturas variables."
    )
    parser.add_argument(
        "--input-dir",
        default="data/facturas",
        help="Carpeta con facturas .txt para evaluar.",
    )
    parser.add_argument(
        "--pattern",
        default="*.txt",
        help="Patrón de archivos a procesar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de evaluación: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos con patrón {args.pattern} en {input_dir}")

    summaries: list[dict] = []
    for invoice_path in files:
        orchestrator = GlobalFlowOrchestrator()
        raw_text = invoice_path.read_text(encoding="utf-8")
        result = orchestrator.run(raw_invoice_text=raw_text, file_name=invoice_path.name)
        observability = result["observability"]
        summaries.append(
            {
                "archivo": invoice_path.name,
                "estado": result["final_result"]["estado"],
                "codigo_final": result["final_result"]["codigo_final"],
                "precision_real": observability["precision_real"],
                "precision_estimada": observability["precision_estimada"],
                "consistencia_score": observability["consistencia_score"],
                "duracion_total_ms": observability["duracion_total_ms"],
                "fallas": len(observability["fallas_detectadas"]),
                "senales_seguridad": len(observability["security_signals"]),
            }
        )

    dashboard_path = generate_dashboard()
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"\nDashboard de observabilidad: {dashboard_path}")


if __name__ == "__main__":
    main()
