#!/usr/bin/env python
# encoding: UTF8

import pandas as pd
from bcnz.paths import input_path

def vipers(engine, quality_cut=False, catalogue_path=None):
    import bcnz

    # This data should have been available in PAUdm.
    # Note: It looks like this actually is ingested by now...
    path = input_path(catalogue_path, "BCNZ_VIPERS_CATALOGUE")
    df_in = pd.read_csv(path, comment='#')

    if quality_cut:
        df_in = df_in[(3 <= df_in.zflg) & (df_in.zflg <= 4)]

    df = df_in.rename(columns={'alpha': 'ra', 'delta': 'dec'})

    # Selecting W1 field.
    df = df[df.ra < 50]

    parent_cat = bcnz.data.paudm_cfhtlens(engine, 'w1')
    specz = bcnz.data.match_position(parent_cat, df)

    return specz
