# AutoArray JAX FFT crop fix

The active `autolens-jax` environment was patched on 2026-08-11 at:

`$CONDA_PREFIX/lib/python3.11/site-packages/autoarray/structures/arrays/kernel_2d.py`

The original environment file is preserved locally as `kernel_2d.py.before`.
After recreating or upgrading the environment, apply the idempotent installer:

```bash
conda run -n autolens-jax python autoarray_jax_crop_patch/apply_autoarray_crop_fix.py
```

`autoarray_jax_crop_fix.patch` is also retained for review and upstream
reporting.  The installer handles either LF or CRLF package source files and
creates a sibling backup before changing an unpatched installation.

The fix changes both JAX FFT crop call sites and corrects the inclusive mask
bounding-box size.  System-8 validation is documented in
`../system_8/JAX_CROP_FIX_TEST.md`.
