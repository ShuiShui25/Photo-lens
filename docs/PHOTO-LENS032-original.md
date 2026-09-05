# PHOTO-LENS032 stable photo-z release

This directory derives from the frozen PHOTO-LENS031 photometric-redshift branch with:

- exact AutoLens `FitImaging` lens-light-subtracted products for source detection;
- source SExtractor `DETECT_THRESH = 4.0 sigma`;
- hysteresis growth level `0.75 * DETECT_THRESH`;
- point-reflection source-fraction overlap classification at `r_s >= 0.4`;
- PAUS NB antisymmetry-centre correction accepted only below 0.15 arcsec;
- `w_NB = 1`;
- HSC systems fitted with PAUS NB + official HSC-G + HSC r/i/z/y;
- CFHT systems fitted with PAUS NB + CFHT u/g/r/i/z.

PHOTO-LENS032 changes only system 3110: NB535 is excluded from both the lens
and source BCNz likelihood. Its measured flux is retained in the photometry
tables for auditability, but it is absent from `common_photoz_bands.csv`, the
signed-pull tables and the final p(z)/flux plots. No lens modeling or aperture
photometry was rerun. Every other system is unchanged from PHOTO-LENS031.

`photometric-estimation/results/` is a copied result snapshot with the single
documented 3110 photo-z update above.
`region-detection/stable-regions/` contains the corresponding automatic lens,
source and point-reflected background regions.  Large input images, AutoLens
model outputs and the 1.2-GB base BCNz cache are not duplicated; their source
release is the sibling `../PHOTO-LENS03` directory.

The combined report is
[`photometric-estimation/all_systems_photoz_report.html`](photometric-estimation/all_systems_photoz_report.html).

## Probability definitions

The physically relevant ordering probability is

\[
P(z_l < z_s) = \int dz_l\,p_l(z_l)\int_{z_l}^{\infty}p_s(z_s)\,dz_s.
\]

For the requested literal inequality statistic, exact inequality of two
continuous variables would have probability one.  We therefore report the
explicitly grid-dependent discrete BCNz definition

\[
P_{\rm grid}(z_l \ne z_s)=1-\sum_i q_l(z_i)q_s(z_i),
\]

where each `q` is the normalized probability mass on the BCNz redshift grid.
This number should not replace `P(z_l < z_s)` for lens validation.
