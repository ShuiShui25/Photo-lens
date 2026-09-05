"""Regression checks for the vendored scientific fixes."""
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import bcnz
from bcnz.fit.libpzqual import get_pzcat

def check_local_fixes():
    root = Path(__file__).resolve().parents[1]
    installed = Path(bcnz.__file__).parent
    for name in ['fit/photoz.py', 'fit/libpzqual.py', 'data/paudm_coadd.py']:
        assert (installed / name).read_bytes() == (root / 'vendor/bcnz/bcnz' / name).read_bytes(), name
    assert (installed / 'data/photoz/all_filters/v4/subaru_g.csv').is_file()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'flux.csv'
        pd.DataFrame({'ref_id':[1], 'band':['NB455'], 'flux':[8.0], 'flux_error':[2.0]}).to_csv(path,index=False)
        loaded = bcnz.data.load_coadd_file(path)
        assert loaded.loc[1, ('flux','pau_nb455')] == 8.0
        assert loaded.loc[1, ('flux_error','pau_nb455')] == 2.0
    z = np.linspace(.01,1.01,101)
    values = np.stack([((z-.31)/.08)**2, ((z-.71)/.08)**2+5.0])[:,None,:]
    chi = xr.DataArray(values, dims=['run','ref_id','z'],coords={'run':[0,1],'ref_id':[1],'z':z})
    cat, pz = get_pzcat(chi, .01, .01)
    shifted, shifted_pz = get_pzcat(chi + 1e6, .01, .01)
    np.testing.assert_allclose(pz, shifted_pz, rtol=1e-8,atol=1e-10)
    np.testing.assert_allclose(pz.sum('z'),1.0)
    assert cat.loc[1,'best_run'] == 0 and shifted.loc[1,'best_run'] == 0
    np.testing.assert_allclose(cat.loc[1,'zb'],.31)
