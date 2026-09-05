# Validation — PHOTO-LENS032 / v0.3.2

Validated on Linux x86_64, Python 3.10.19. Scientific code commit:
`612d9ea5295e5bc1def514d73943af7fa8f43dbb`.

| Check | Result | Scope |
| --- | --- | --- |
| Fresh isolated environment, install requirements.txt | PASS | Local BCNz wheel and both Cython extensions; packaged filter/SED assets |
| pip check | PASS | No broken requirements |
| Real-image integration test | PASS | System 12, 45 bands, 21/41 subpixels, all model runs with redshift stride 20 |
| Local BCNz regression checks | PASS | No implicit NB rescaling; peak best-run; likelihood underflow; installed modified source |
| Native system-12 recomputation | PASS | Original images, frozen regions, 81/161 subpixels, full redshift grid; 246.8 s |
| Native system-3110 refit | PASS | Frozen measured photometry, official HSC-G, 39 NB + 5 BB, NB535 excluded |
| Nine-system stable result hashes | PASS | Original PHOTO-LENS032 results unchanged |

Native-grid p(z) arrays for lens and source were compared with the frozen
PHOTO-LENS032 parquet outputs. All four arrays (systems 12 and 3110) have
**maximum absolute difference 0.0**. Peak redshifts: system 12 lens/source
0.447/0.243; system 3110 lens/source 0.643/0.073. See
[native-validation.json](native-validation.json).

The primary integration test uses the real PAUS/CFHT input images and frozen
DS9 regions shipped in Git. It checks source/lens photometry, noise and mask
convergence, BCNz fits, posterior validity, output pulls, the 3110 exclusion
rule, and the scientific baseline checksums. It does not require a PAU database,
an external BCNz checkout, or full science caches.

The full-resolution native checks used separately imported PHOTO-LENS03 assets;
they do not establish that the complete nine-system science dataset is in Git.
GPU AutoLens fitting and fresh SExtractor detection were not rerun. The Conda
YAML and original GPU package inventory are supplied, but a fresh GPU/Conda
environment was not independently validated; the tested installation is venv
plus the locked pip requirements.

## GitHub clone verification

**PASS**: `scripts/verify_from_git.sh` created a second fresh venv, cloned
`git@github.com:ShuiShui25/Photo-lens.git`, installed `requirements.txt` from
that clone (including compilation of its vendored BCNz), ran `pip check`, and
completed `tests/smoke_test.py` in **30.36 seconds**.

Verified GitHub commit: `612d9ea5295e5bc1def514d73943af7fa8f43dbb`.
The stable tag adds only validation documentation to this tested code.

BCNz was loaded from the new environment:
`/tmp/photo-lens-clone-6t5hFh/env/lib/python3.10/site-packages/bcnz/__init__.py`.
The checkout and environment remain in `/tmp/photo-lens-clone-6t5hFh` on the
validation machine. Download used the machine's existing SSH proxy; no prior
installed scientific packages or outside science-data trees were used by the
clone test. Cached package downloads were permitted.

Evidence: [clone result JSON](validation/github-clone-result.json),
[complete clone/install/test log](validation/github-clone.log),
[native image test log](validation/native-system12.log), and
[3110 refit log](validation/native-system3110-refit.log).
