#!/usr/bin/env python3
"""Real FITS -> aperture photometry -> BCNz -> p(z)/plots integration test.

Uses actual system-12 images and templates with a thinned z grid. Does not
rerun GPU lens modeling or SExtractor; those stages supplied frozen regions.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'photometric-estimation'))
import numpy as np
import pandas as pd
import bcnz
import mc_genz_cython
import prior_volume_integral
from paus_lensing034 import PipelineConfig, run_pipeline
import paus_lensing034.config as config_module
from paus_lensing034.photoz import _common_catalogues


def main():
    from bcnz_regressions import check_local_fixes
    check_local_fixes()
    fixture = ROOT / 'tests/fixtures/system12'
    manifest = json.loads((fixture / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((fixture / name).read_bytes()).hexdigest()
        assert actual == expected, f'Fixture changed: {name}'
    # Verify all frozen scientific results remain exactly the release snapshot.
    for name, expected in json.loads((ROOT / 'docs/stable-results-sha256.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    with tempfile.TemporaryDirectory(prefix='photo-lens-test-') as tmp:
        project = Path(tmp)
        shutil.copytree(fixture, project, dirs_exist_ok=True)
        config_module.PROJECT_ROOT = project
        cfg = PipelineConfig(lens_id=12, aperture_subpixels=21, convergence_subpixels=41, nb_aperture_centre_table=project / 'photometric-estimation/nb-centres/system_12/system12_nb_antisymmetry_centres.csv')
        result = run_pipeline(cfg)
        assert len(result.common_bands) >= 30, result.common_bands
        assert set(result.summary.target) == {'lens', 'source'}
        for target in ['lens', 'source']:
            pz = pd.read_parquet(result.result_dir / f'wNB1/pzcat_{target}.pq')
            columns = [c for c in pz if c.startswith('z') and c[1:].isdigit()]
            probability = pz[columns].to_numpy(float)
            assert np.isfinite(probability).all() and (probability >= 0).all()
            assert (probability.sum(axis=1) > 0).all()
            pulls = pd.read_csv(result.result_dir / f'wNB1/signed_pulls_{target}.csv')
            measured = ((pulls.flux - pulls.model_flux) / pulls.flux_error).to_numpy()
            np.testing.assert_allclose(measured, pulls.pull, rtol=1e-8, atol=1e-8)
        # NB535 must be removed only for 3110, without deleting measured flux.
        class SelectionConfig:
            ordered_bands = ['NB525', 'NB535', 'subaru_i']
            bb_names = ['subaru_i']
            bb_survey = 'HSC'
            excluded_bands = PipelineConfig(lens_id=3110).excluded_bands
        table = pd.DataFrame({'band': SelectionConfig.ordered_bands, 'flux': [1.,2.,3.], 'flux_error': [1.,1.,1.]})
        common, _, status = _common_catalogues(SelectionConfig(), table, table)
        assert 'NB535' not in common and len(table) == 3
        assert 'fixed exclusion' in status.set_index('band').loc['NB535', 'exclusion_reason']
        output = ROOT / '.validation/smoke'
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(result.result_dir, output / '12', dirs_exist_ok=True)
        report = {'status':'passed', 'python':sys.version, 'bcnz':bcnz.__file__,
                  'system':12, 'bands':len(result.common_bands), 'elapsed_seconds':result.elapsed_seconds,
                  'scope':'real image photometry and BCNz; frozen regions; z stride 20; no GPU modeling'}
        (output / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
