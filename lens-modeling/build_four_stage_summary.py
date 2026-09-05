#!/usr/bin/env python3
"""Build the reproducible PHOTO-LENS03 four-stage modeling summary."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = ROOT / "lens-modeling"
SYSTEMS = ("2", "8", "12", "3110", "3336", "3441", "13369", "12971", "13246")


def notebook_errors(path: Path) -> int:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        output.get("output_type") == "error"
        for cell in notebook.get("cells", [])
        for output in cell.get("outputs", [])
    )


def main() -> None:
    selection = json.loads(
        (MODEL_ROOT / "bb_selection_manifest.json").read_text(encoding="utf-8")
    )
    rows = []
    manifest = {}

    for system in SYSTEMS:
        output = MODEL_ROOT / f"system_{system}" / "output"
        summary_path = output / "stage4_run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        notebook = MODEL_ROOT / f"{system}_four_stage_modeling.executed.ipynb"
        previews = [output / "stage_previews" / f"stage{i}_fit.png" for i in range(1, 5)]
        wcs = output / "wcs_products_stage4" / f"{system}_wcs_product_manifest.json"
        errors = notebook_errors(notebook)
        complete = (
            errors == 0
            and all(path.is_file() for path in previews)
            and wcs.is_file()
            and summary.get("jax_platform") == "gpu"
            and summary.get("source", {}).get("regularization")
            == "ConstantPlusBrightnessZeroth"
            and summary.get("source", {}).get("signal_scale") == 0.5
            and summary.get("stage4_regularization_revision") == "clipsig1"
        )
        item = {
            "system": system,
            "survey": selection[system]["survey"],
            "band": selection[system]["band"],
            "selection_reason": selection[system]["reason"],
            "stage4_log_evidence": summary["log_evidence"],
            "stage4_reduced_chi_squared": summary["reduced_chi_squared"],
            "jax_platform": summary["jax_platform"],
            "regularization": summary["source"]["regularization"],
            "signal_scale": summary["source"]["signal_scale"],
            "regularization_revision": summary["stage4_regularization_revision"],
            "executed_notebook": str(notebook.relative_to(ROOT)),
            "notebook_error_cells": errors,
            "stage4_wcs_manifest": str(wcs.relative_to(ROOT)),
            "validated": complete,
        }
        manifest[system] = item
        preview_html = "".join(
            f'<a href="{html.escape(str(path.relative_to(MODEL_ROOT)))}">'
            f'<img src="{html.escape(str(path.relative_to(MODEL_ROOT)))}" alt="stage {i}"></a>'
            for i, path in enumerate(previews, 1)
        )
        rows.append(
            "<tr>"
            f"<td><strong>{system}</strong></td>"
            f"<td>{html.escape(item['survey'])}</td>"
            f"<td>{html.escape(item['band'])}</td>"
            f"<td>{html.escape(item['selection_reason'])}</td>"
            f"<td>{item['stage4_log_evidence']:.3f}</td>"
            f"<td>{item['stage4_reduced_chi_squared']:.3f}</td>"
            f"<td>{'PASS' if complete else 'FAIL'}</td>"
            f'<td><a href="{notebook.name}">executed notebook</a></td>'
            f'<td><div class="previews">{preview_html}</div></td>'
            "</tr>"
        )

    (MODEL_ROOT / "four_stage_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PHOTO-LENS03 four-stage modeling summary</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#202124;background:#f7f8fa}}
h1{{margin-bottom:4px}} .note{{max-width:1100px;color:#4b5563}}
table{{border-collapse:collapse;width:100%;background:white;box-shadow:0 1px 5px #ccd}}
th,td{{border:1px solid #d8dce2;padding:8px;vertical-align:top;text-align:left}}
th{{background:#eef2f7;position:sticky;top:0}} .previews{{display:flex;gap:4px;min-width:420px}}
.previews img{{width:102px;height:102px;object-fit:contain;background:#fff}}
tr:nth-child(even){{background:#fafcff}} a{{color:#075985}}
</style></head><body>
<h1>PHOTO-LENS03 four-stage modeling summary</h1>
<p class="note">BB is selected automatically: a complete, position-compatible HSC tuple is preferred; otherwise CFHT is used. All listed stage-4 fits use JAX-GPU, a fixed 20x20 pixel source, and ConstantPlusBrightnessZeroth with signal_scale=0.5 and the clipped [0,1] adapt-signal domain fix (<code>clipsig1</code>).</p>
<table><thead><tr><th>System</th><th>Survey</th><th>Band</th><th>Selection</th><th>log Z</th><th>reduced χ²</th><th>Validation</th><th>Notebook</th><th>Stages 1–4</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    (MODEL_ROOT / "four_stage_modeling_summary.html").write_text(
        document, encoding="utf-8"
    )
    if not all(item["validated"] for item in manifest.values()):
        raise SystemExit("At least one system failed final validation")
    print(f"Validated {len(manifest)} systems; summary written to {MODEL_ROOT}")


if __name__ == "__main__":
    main()
