from __future__ import annotations

import argparse
import json
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from app.config import project_root, settings


def observability_dir() -> Path:
    path = project_root() / settings.observability_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def pct(value: float | None) -> str:
    if value is None:
        return "sin etiqueta"
    return f"{value * 100:.1f}%"


def avg(values: list[float]) -> float:
    return mean(values) if values else 0.0


def status_class(value: float, good_threshold: float, warn_threshold: float) -> str:
    if value >= good_threshold:
        return "good"
    if value >= warn_threshold:
        return "warn"
    return "bad"


def build_cards(metrics: list[dict[str, Any]]) -> str:
    total = len(metrics)
    approved = sum(1 for item in metrics if item.get("estado_final") == "aprobado")
    human_review = sum(1 for item in metrics if item.get("requiere_revision_humana"))
    estimated_precision = avg([float(item.get("precision_estimada", 0)) for item in metrics])
    real_values = [float(item["precision_real"]) for item in metrics if item.get("precision_real") is not None]
    real_precision = avg(real_values) if real_values else None
    consistency = avg([float(item.get("consistencia_score", 0)) for item in metrics])
    latency = avg([float(item.get("duracion_total_ms", 0)) for item in metrics])

    cards = [
        ("Ejecuciones", str(total), "Total de corridas observadas", "neutral"),
        ("Aprobadas", str(approved), f"{human_review} enviadas a revisión humana", "good" if approved else "neutral"),
        ("Precisión real", pct(real_precision), "Usa ground_truth.json cuando existe", "good" if real_precision and real_precision >= 0.85 else "warn"),
        ("Precisión estimada", pct(estimated_precision), "Proxy por confianza, evidencia y validación", status_class(estimated_precision, 0.85, 0.70)),
        ("Consistencia", pct(consistency), "Alineación clasificador/validador/reglas", status_class(consistency, 0.90, 0.75)),
        ("Latencia media", f"{latency:.0f} ms", "Tiempo promedio end-to-end", "neutral"),
    ]

    return "\n".join(
        f"""
        <article class="card {css_class}">
          <span>{escape(title)}</span>
          <strong>{escape(value)}</strong>
          <small>{escape(description)}</small>
        </article>
        """
        for title, value, description, css_class in cards
    )


def build_latency_table(metrics: list[dict[str, Any]]) -> str:
    by_step: dict[str, list[float]] = {}
    for item in metrics:
        for step, latency in item.get("latencia_por_paso_ms", {}).items():
            by_step.setdefault(step, []).append(float(latency))

    if not by_step:
        return "<p class='empty'>No hay latencias registradas todavía.</p>"

    max_latency = max(avg(values) for values in by_step.values()) or 1.0
    rows = []
    for step, values in sorted(by_step.items(), key=lambda item: avg(item[1]), reverse=True):
        average = avg(values)
        width = max(4, min(100, average / max_latency * 100))
        rows.append(
            f"""
            <tr>
              <td>{escape(step)}</td>
              <td>{average:.0f} ms</td>
              <td><div class="bar"><i style="width:{width:.1f}%"></i></div></td>
            </tr>
            """
        )
    return f"<table><thead><tr><th>Paso</th><th>Latencia media</th><th>Comparación</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_recent_runs(metrics: list[dict[str, Any]]) -> str:
    if not metrics:
        return "<p class='empty'>Ejecuta primero <code>python -m app.main data/facturas/factura_demo.txt</code>.</p>"

    rows = []
    for item in reversed(metrics[-12:]):
        signals = item.get("security_signals", [])
        failures = item.get("fallas_detectadas", [])
        rows.append(
            f"""
            <tr>
              <td>{escape(item.get("archivo", ""))}</td>
              <td>{escape(item.get("estado_final", ""))}</td>
              <td>{pct(item.get("precision_real"))}</td>
              <td>{pct(float(item.get("precision_estimada", 0)))}</td>
              <td>{pct(float(item.get("consistencia_score", 0)))}</td>
              <td>{float(item.get("duracion_total_ms", 0)):.0f} ms</td>
              <td>{len(failures)}</td>
              <td>{len(signals)}</td>
            </tr>
            """
        )

    return (
        "<table><thead><tr><th>Archivo</th><th>Estado</th><th>Precisión real</th>"
        "<th>Precisión estimada</th><th>Consistencia</th><th>Latencia</th><th>Fallas</th><th>Seguridad</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def build_recommendations(metrics: list[dict[str, Any]]) -> str:
    counter: dict[str, int] = {}
    for item in metrics:
        for recommendation in item.get("recomendaciones", []):
            counter[recommendation] = counter.get(recommendation, 0) + 1

    if not counter:
        return "<p class='empty'>No hay recomendaciones acumuladas todavía.</p>"

    rows = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    return "\n".join(
        f"<li><strong>{count}x</strong> {escape(recommendation)}</li>"
        for recommendation, count in rows[:10]
    )


def build_failures(metrics: list[dict[str, Any]]) -> str:
    counter: dict[str, int] = {}
    for item in metrics:
        for failure in item.get("fallas_detectadas", []):
            counter[failure] = counter.get(failure, 0) + 1

    if not counter:
        return "<p class='empty'>Sin fallas detectadas en las ejecuciones registradas.</p>"

    return "\n".join(
        f"<li><strong>{count}x</strong> {escape(failure)}</li>"
        for failure, count in sorted(counter.items(), key=lambda item: item[1], reverse=True)
    )


def render_dashboard(metrics: list[dict[str, Any]]) -> str:
    generated_at = escape(str(Path(settings.observability_dir) / "metrics.jsonl"))
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GlobalFlow Observabilidad</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #64748b;
      --line: #d9e0e8;
      --blue: #2563eb;
      --green: #16803c;
      --amber: #b7791f;
      --red: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: 28px; letter-spacing: 0; }}
    header p {{ margin: 8px 0 0; color: var(--muted); }}
    main {{ padding: 24px 36px 40px; }}
    section {{ margin-bottom: 24px; }}
    h2 {{ font-size: 18px; margin: 0 0 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card span {{ display: block; color: var(--muted); font-size: 13px; }}
    .card strong {{ display: block; margin: 8px 0; font-size: 26px; }}
    .card small {{ color: var(--muted); line-height: 1.35; }}
    .card.good {{ border-left: 4px solid var(--green); }}
    .card.warn {{ border-left: 4px solid var(--amber); }}
    .card.bad {{ border-left: 4px solid var(--red); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      overflow-x: auto;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: middle; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .bar {{ width: 100%; min-width: 160px; height: 10px; background: #e8edf3; border-radius: 999px; overflow: hidden; }}
    .bar i {{ display: block; height: 100%; background: var(--blue); }}
    ul {{ margin: 0; padding-left: 20px; }}
    li {{ margin: 8px 0; line-height: 1.45; }}
    code {{ background: #eef2f7; border-radius: 4px; padding: 2px 5px; }}
    .empty {{ color: var(--muted); margin: 0; }}
    @media (max-width: 720px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      h1 {{ font-size: 23px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>GlobalFlow Observabilidad</h1>
    <p>Métricas de precisión, latencia, consistencia, trazabilidad, seguridad y recomendaciones. Fuente: <code>{generated_at}</code></p>
  </header>
  <main>
    <section class="grid">
      {build_cards(metrics)}
    </section>

    <section>
      <h2>Latencia por Paso</h2>
      <div class="panel">{build_latency_table(metrics)}</div>
    </section>

    <section>
      <h2>Ejecuciones Recientes</h2>
      <div class="panel">{build_recent_runs(metrics)}</div>
    </section>

    <section>
      <h2>Puntos de Falla</h2>
      <div class="panel"><ul>{build_failures(metrics)}</ul></div>
    </section>

    <section>
      <h2>Recomendaciones de Optimización</h2>
      <div class="panel"><ul>{build_recommendations(metrics)}</ul></div>
    </section>
  </main>
</body>
</html>
"""


def generate_dashboard(output_path: Path | None = None) -> Path:
    base_dir = observability_dir()
    metrics = read_jsonl(base_dir / "metrics.jsonl")
    output = output_path or base_dir / "dashboard.html"
    output.write_text(render_dashboard(metrics), encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera dashboard HTML de observabilidad GlobalFlow.")
    parser.add_argument("--output", default=None, help="Ruta de salida del dashboard HTML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_dashboard(Path(args.output) if args.output else None)
    print(f"Dashboard generado: {output}")


if __name__ == "__main__":
    main()
