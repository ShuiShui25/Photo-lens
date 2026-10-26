"""Run with the photometry environment: python -m unittest discover -s tests."""
import contextlib
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import bcnz
import pandas as pd
from bcnz.paths import input_path

ROOT = Path(__file__).resolve().parents[1]


class PortableInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bcnz paths ")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.environment = patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        for key in list(os.environ):
            if key.startswith("BCNZ_"):
                os.environ.pop(key)

    def test_explicit_path_takes_precedence_and_missing_inputs_explain_configuration(self):
        source = self.folder / "input catalogue.csv"
        source.write_text("id,z\n1,0.3\n")
        os.environ["BCNZ_TEST_INPUT"] = str(self.folder / "missing")
        self.assertEqual(input_path(source, "BCNZ_TEST_INPUT"), source)
        with self.assertRaisesRegex(FileNotFoundError, "BCNZ_TEST_INPUT"):
            input_path(None, "BCNZ_TEST_INPUT")
        os.environ.pop("BCNZ_TEST_INPUT")
        with self.assertRaisesRegex(ValueError, "BCNZ_TEST_INPUT"):
            input_path(None, "BCNZ_TEST_INPUT")
        with self.assertRaises(FileNotFoundError):
            input_path(self.folder, "BCNZ_TEST_INPUT")
        self.assertEqual(input_path(self.folder, "BCNZ_TEST_INPUT", directory=True), self.folder)

    def test_relative_environment_and_home_expansion(self):
        source = self.folder / "catalogue.csv"
        source.touch()
        relative = os.path.relpath(source)
        os.environ["BCNZ_TEST_INPUT"] = relative
        self.assertEqual(input_path(None, "BCNZ_TEST_INPUT"), source)
        self.assertEqual(input_path('~', 'BCNZ_TEST_INPUT', directory=True), Path.home().resolve())

    def test_optional_configs_require_external_data_and_support_pandas_two(self):
        for configure in [bcnz.config.test_bayevz, bcnz.config.pauscosmos_deep]:
            with self.subTest(configure=configure.__name__):
                with self.assertRaisesRegex(ValueError, "BCNZ_BAYEVZ_DATA_DIR"):
                    configure()
        for directory in ['filt_70bands_noirach', 'seds', 'ext_laws', 'calib_filters']:
            (self.folder / directory).mkdir()
        os.environ['BCNZ_BAYEVZ_DATA_DIR'] = str(self.folder)
        for configure in [bcnz.config.test_bayevz, bcnz.config.pauscosmos_deep]:
            explicit = configure(data_dir=self.folder)
            fallback = configure()
            self.assertGreater(len(explicit), 0)
            self.assertEqual(len(explicit), len(fallback))
            self.assertEqual(set(explicit.sed_dir), {str(self.folder / 'seds')})
            self.assertEqual(set(explicit.filter_dir), {str(self.folder / 'filt_70bands_noirach')})
        (self.folder / 'seds').rmdir()
        with self.assertRaisesRegex(FileNotFoundError, 'seds'):
            bcnz.config.test_bayevz()

    def test_production_templates_remain_bundled(self):
        runs = bcnz.config.eriksen2019()
        self.assertEqual(len(runs), 35)
        for name in runs.sed_dir.unique():
            self.assertTrue(Path(name).is_dir())
            self.assertTrue(Path(name).is_relative_to(Path(bcnz.__file__).parent))

    def test_specz_fails_before_database_access_when_unconfigured(self):
        for module, function in [('sdss', 'sdss'), ('sdss_gama', 'sdss_gama'), ('vipers', 'vipers')]:
            mod = importlib.import_module('bcnz.specz.' + module)
            with self.subTest(function=function), self.assertRaisesRegex(ValueError, 'BCNZ_'):
                getattr(mod, function)(None)

    def test_sdss_catalogue_parameter_preserves_column_mapping(self):
        mod = importlib.import_module('bcnz.specz.sdss')
        source = self.folder / 'sdss.csv'
        source.write_text('metadata\nredshift,zwarning,class,i\n0.3,0,GALAXY,20\n')
        with patch.object(mod, '_query_refcat', return_value=pd.DataFrame()), \
             patch('bcnz.data.match_position', side_effect=lambda parent, catalogue: catalogue), \
             contextlib.redirect_stdout(None):
            result = mod.sdss(None, catalogue_path=source)
        self.assertEqual(result.zspec.tolist(), [0.3])
        self.assertEqual(result.obj_type.tolist(), ['GALAXY'])

    def test_sdss_gama_environment_preserves_quality_cuts(self):
        mod = importlib.import_module('bcnz.specz.sdss_gama')
        source = self.folder / 'gama.csv'
        source.write_text('RA,DEC,Z,nQ,objid\n130,0,0.3,4,1\n131,0,0.4,2,2\n132,0,0,4,3\n')
        os.environ['BCNZ_SDSS_GAMA_CATALOGUE'] = str(source)
        with patch.object(mod, '_query_refcat', return_value=pd.DataFrame()), \
             patch('bcnz.data.match_position', side_effect=lambda parent, catalogue: catalogue):
            result = mod.sdss_gama(None)
        self.assertEqual(result.objid.tolist(), [1])
        self.assertEqual(result.zspec.tolist(), [0.3])

    def test_vipers_environment_preserves_positional_quality_flag_and_field_cut(self):
        source = self.folder / 'vipers.csv'
        source.write_text('alpha,delta,zflg\n36,-4,3\n37,-4,2\n100,0,4\n')
        os.environ['BCNZ_VIPERS_CATALOGUE'] = str(source)
        with patch('bcnz.data.paudm_cfhtlens', return_value=pd.DataFrame()), \
             patch('bcnz.data.match_position', side_effect=lambda parent, catalogue: catalogue):
            result = bcnz.specz.vipers(None, True)
            unfiltered = bcnz.specz.vipers(None, catalogue_path=source)
        self.assertEqual(result.ra.tolist(), [36])
        self.assertEqual(unfiltered.ra.tolist(), [36, 37])

    def test_bayevz_cli_exposes_paths_without_requiring_mpi_for_help(self):
        result = subprocess.run([sys.executable, str(ROOT / 'vendor/bcnz/bcnz/bin/run_bayevz_simple.py'), '--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in ['--model-dir', '--output-dir', '--data-dir', '--catalogue-file']:
            self.assertIn(option, result.stdout)

    def test_real_sextractor_handles_spaced_and_relative_work_directories(self):
        import numpy as np
        from astropy.io import fits
        from astropy.wcs import WCS
        spec = importlib.util.spec_from_file_location('portable_regions', ROOT / '1.lens-modeling/region_selection/pipeline.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        try:
            executable = module.find_sextractor(os.environ.get('PHOTO_LENS_SEXTRACTOR'))
        except RuntimeError as exc:
            self.skipTest(str(exc))
        y, x = np.indices((64, 64))
        signal = 50 * np.exp(-((x - 32)**2 + (y - 32)**2) / 18)
        signal += np.random.default_rng(7).normal(0, 1, signal.shape)
        product = module.Product(signal, fits.Header(), WCS(naxis=2),
                                 np.ones(signal.shape, dtype=bool), 0, 0, 32, 32, 0, 1)
        base = {'detect_thresh': 3, 'detect_snr_floor': 3, 'detect_peak_fraction': 0.1,
                'detect_minarea': 5, 'deblend_nthresh': 32, 'deblend_mincont': 0.001,
                'back_size': 16}
        for mode in ['raw', 'snr']:
            settings = dict(base, input_mode=mode)
            absolute = self.folder / mode / 'work with spaces'
            relative = Path(os.path.relpath(self.folder / mode / 'relative work'))
            first = module.run_sextractor(product, absolute, executable, settings)
            second = module.run_sextractor(product, relative, executable, settings)
            self.assertGreater(np.count_nonzero(first), 0)
            np.testing.assert_array_equal(first, second)

    def test_modeling_cache_defaults_follow_checkout(self):
        # Import in a fresh process so prior Matplotlib/JAX state cannot mask the defaults.
        code = '''
import importlib.util, os, sys
from pathlib import Path
root=Path(sys.argv[1]); expected=str(root / '.runtime/matplotlib')
for relative in ['1.lens-modeling/run_three_stage.py', '1.lens-modeling/region_selection/pipeline.py']:
    os.environ.pop('MPLCONFIGDIR', None)
    spec=importlib.util.spec_from_file_location('cache_check', root / relative)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    if hasattr(mod, 'configure_jax'): mod.configure_jax('cpu')
    assert os.environ['MPLCONFIGDIR'] == expected, os.environ['MPLCONFIGDIR']
'''
        result = subprocess.run([sys.executable, '-c', code, str(ROOT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
