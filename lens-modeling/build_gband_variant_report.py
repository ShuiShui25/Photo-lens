#!/usr/bin/env python3
"""Compare preserved baseline fits with the non-destructive G-band variants."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SYSTEMS = ("8", "12", "3336")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    manifest = {}
    for system in SYSTEMS:
        system_root = ROOT / f"system_{system}"
        base_cfg = load(system_root / "config.json")
        g_cfg = load(system_root / "config_gband.json")
        base = load(system_root / "output" / "run_summary.json")
        g = load(system_root / "output_gband" / "run_summary.json")
        base4 = load(system_root / "output" / "stage4_run_summary.json")
        g4 = load(system_root / "output_gband" / "stage4_run_summary.json")
        bchi = [base["reduced_chi_squared"][f"stage{i}"] for i in range(1, 4)]
        gchi = [g["reduced_chi_squared"][f"stage{i}"] for i in range(1, 4)]
        bchi.append(base4["reduced_chi_squared"])
        gchi.append(g4["reduced_chi_squared"])
        manifest[system] = {
            "baseline_band": base_cfg["bb_band"],
            "g_band": g_cfg["bb_band"],
            "baseline_reduced_chi_squared": dict(zip(("stage1", "stage2", "stage3", "stage4"), bchi)),
            "gband_reduced_chi_squared": dict(zip(("stage1", "stage2", "stage3", "stage4"), gchi)),
            "gband_regularization": g4["source"]["regularization"],
            "gband_signal_scale": g4["source"]["signal_scale"],
            "gband_jax_platform": g4["jax_platform"],
        }
        cells = "".join(f"<td>{x:.3f}</td>" for pair in zip(bchi, gchi) for x in pair)
        images = "".join(
            f'<figure><img src="system_{system}/output_gband/stage_previews/stage{i}_fit.png">'
            f'<figcaption>G stage {i}</figcaption></figure>' for i in range(1, 5)
        )
        rows.append(
            f"<tr><td><strong>{system}</strong></td><td>{base_cfg['bb_band']}</td>"
            f"<td>{g_cfg['bb_band']}</td>{cells}"
            f'<td><a href="{system}_gband_four_stage_modeling.executed.ipynb">notebook</a></td></tr>'
            f'<tr><td colspan="12"><div class="images">{images}</div></td></tr>'
        )
    (ROOT / "gband_variant_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PHOTO-LENS03 G-band variants</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f7f9;color:#202124}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ccd;padding:8px;text-align:left}}
th{{background:#edf2f7}}.images{{display:flex;gap:10px}}figure{{margin:0;flex:1}}
img{{width:100%;max-height:330px;object-fit:contain}}figcaption{{text-align:center}}a{{color:#075985}}
</style></head><body><h1>G-band four-stage variants</h1>
<p>The original outputs are preserved in <code>output</code>; variants are in <code>output_gband</code>. Reduced χ² is shown for within-system diagnostics. Log evidence is intentionally not compared across different images/bands.</p>
<table><thead><tr><th rowspan="2">System</th><th rowspan="2">Baseline</th><th rowspan="2">Variant</th>
<th colspan="2">Stage 1 χ²ν</th><th colspan="2">Stage 2 χ²ν</th><th colspan="2">Stage 3 χ²ν</th><th colspan="2">Stage 4 χ²ν</th><th rowspan="2">Result</th></tr>
<tr><th>base</th><th>G</th><th>base</th><th>G</th><th>base</th><th>G</th><th>base</th><th>G</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    (ROOT / "gband_variant_comparison.html").write_text(html, encoding="utf-8")
    print(ROOT / "gband_variant_comparison.html")


if __name__ == "__main__":
    main()
