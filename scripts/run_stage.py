#!/usr/bin/env python3
"""Execute one Photo-lens stage for a named system."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from photo_lens.workflow import system_record
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.runtime/matplotlib'))
os.environ.setdefault('JAX_ENABLE_X64', 'true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

def command(*args):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)

def module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def modeling(name, mode):
    from photo_lens.discovery import ensure_model_config
    record = system_record(name)
    config_path = ensure_model_config(record)
    if not record['has_model']:
        mode = 'fit'
    if mode == "render":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    base = ROOT / '1.lens-modeling'
    command(sys.executable, base / 'autoarray_jax_crop_patch/apply_autoarray_crop_fix.py')
    sys.path.insert(0, str(base))
    if mode == 'fit':
        # Search in a fresh directory. Promote completed outputs only after all
        # four stages succeed, retaining the prior model outside public assets.
        from dataclasses import asdict
        from lens_modeling.config import load_config
        original = load_config(config_path)
        session = ROOT / '.runtime/refits' / f'{name}-{time.time_ns()}'
        session.mkdir(parents=True)
        values = asdict(original)
        values = {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}
        values['output_dir'] = str(session / 'output')
        values['stage4_output_dir'] = None
        search_config = session / 'config.json'
        search_config.write_text(json.dumps(values, indent=2)+'\n')
        platform = os.environ.get('PHOTO_LENS_JAX_PLATFORM', 'gpu')
        os.environ['JAX_PLATFORMS'] = 'cpu' if platform == 'cpu' else 'cuda'
        command(sys.executable, base / 'run_three_stage.py', '--config', search_config, '--jax-platform', platform)
        command(sys.executable, base / 'run_stage4.py', search_config)
        previous = session / 'previous-output'
        if original.output_dir.exists():
            original.output_dir.rename(previous)
        original.output_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            (session / 'output').rename(original.output_dir)
        except Exception:
            if previous.exists():
                previous.rename(original.output_dir)
            raise
    # Re-evaluate actual likelihoods from saved tracers, not image-file copies.
    import autolens as al
    import autolens.plot as aplt
    from lens_modeling.config import load_config
    from lens_modeling.data import load_dataset
    from lens_modeling.regularization import ConstantPlusBrightnessZeroth
    al.reg.ConstantPlusBrightnessZeroth = ConstantPlusBrightnessZeroth
    exporter = module_from(ROOT / '2.region-detection/export_satellite_products.py', 'photo_lens_exact_export')
    config = load_config(config_path)
    bundle = load_dataset(config)
    preview = config.output_dir / 'stage_previews'
    preview.mkdir(parents=True, exist_ok=True)
    records = []
    for stage in range(1, 5):
        tracer = al.from_json(file_path=exporter.stage_tracer(config.output_dir, stage))
        fit = exporter.maximum_likelihood_fit(config, bundle, tracer, stage) if stage >= 3 else al.FitImaging(dataset=bundle.dataset_masked, tracer=tracer)
        aplt.FitImagingPlotter(fit=fit, mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=str(preview), filename=f'stage{stage}_fit', format='png'))).subplot_fit()
        records.append({'stage':stage, 'chi_squared':float(fit.chi_squared), 'reduced_chi_squared':float(fit.reduced_chi_squared)})
        print(f'Stage {stage}: chi-squared={fit.chi_squared:.6f}', flush=True)
    (preview / 'fit_metrics.json').write_text(json.dumps(records, indent=2)+'\n')
    command(sys.executable, ROOT / '2.region-detection/export_satellite_products.py', '--config', config_path, '--products-dir', config.output_dir / 'wcs_products_stage4', '--stage', '4')

def regions(name):
    base = ROOT / '2.region-detection'
    executable = os.environ.get('PHOTO_LENS_SEXTRACTOR') or shutil.which('source-extractor') or shutil.which('sex')
    if not executable:
        raise RuntimeError('SExtractor is required: conda install -c conda-forge source-extractor, or set PHOTO_LENS_SEXTRACTOR.')
    products = ROOT / f'1.lens-modeling/{name}/output/wcs_products_stage4'
    command(sys.executable, base / 'region_selection_v2.py', '--system', name, '--products-dir', products, '--positions-file', ROOT / f'data/bb/{name}/positions_gui.txt', '--background-centre', 'positions-gui', '--lens-scale', '1.0', '--sextractor', executable, '--output-dir', base / 'outputs' / name)
    os.environ['PHOTO_LENS_SEXTRACTOR'] = executable
    command(sys.executable, base / 'apply_point_fraction_selection.py', '--system', name)
    from photo_lens.visuals import plot_regions
    plot_regions(name)

def photoz(name):
    sys.path.insert(0, str(ROOT / '3.photometric-estimation'))
    from photometry_engine import PipelineConfig, run_pipeline
    settings = json.loads((ROOT / 'config/photometry.json').read_text())
    record = system_record(name)
    centres = ROOT / f'3.photometric-estimation/nb-centres/{name}/{name}_nb_antisymmetry_centres.csv'
    if not centres.is_file() and record.get('nb_centre_path'):
        centres = ROOT / record['nb_centre_path']
    if not centres.is_file():
        command(sys.executable, ROOT / '3.photometric-estimation/measure_nb_antisymmetry_centres.py',
                '--system', name, '--output-root', ROOT / '3.photometric-estimation/nb-centres',
                '--max-shift-arcsec', settings['nb_centre_max_shift_arcsec'])
    cfg = PipelineConfig(lens_id=name, field=record['field'], catalog_rel=Path(record['catalogue_path']),
        include_hsc_g=record['bb_survey']=='HSC', aperture_subpixels=settings['aperture_subpixels'],
        convergence_subpixels=settings['convergence_subpixels'], nb_weights=(float(settings['nb_weight']),),
        official_nb_weight=float(settings['nb_weight']), nb_aperture_centre_table=centres)
    result = run_pipeline(cfg)
    from photo_lens.visuals import plot_photoz
    plot_photoz(name)
    print(result.summary.to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['modeling', 'regions', 'photoz'])
    parser.add_argument('--system', default='J022511-045433')
    parser.add_argument('--mode', choices=['render','fit'], default='render')
    args = parser.parse_args()
    system_record(args.system)
    lock_path = ROOT / '.runtime/workflow.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another Photo-lens stage is running. Wait for it to finish before starting a notebook or web task.') from exc
        if args.stage == 'modeling': modeling(args.system, args.mode)
        elif args.stage == 'regions': regions(args.system)
        else: photoz(args.system)
    print(f'Completed {args.stage}: {args.system}', flush=True)
if __name__ == '__main__': main()
