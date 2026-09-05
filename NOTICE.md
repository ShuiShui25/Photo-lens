# Third-party provenance

The repository's original MIT license applies to the author's PHOTO-LENS code.
Vendored BCNz is separately licensed under GPL-3.0-or-later; its license and
original copyright notices are preserved in `vendor/bcnz/LICENCE.TXT` and source
files. The root MIT license does not relicense BCNz. Distribution or use of the
combined software must respect the applicable BCNz license.

BCNz upstream: https://github.com/PAU-survey/bcnz
Local fork: https://github.com/ShuiShui25/bcnz
The imported fork commit is recorded in `vendor/bcnz/UPSTREAM_COMMIT.txt`.
`LOCAL_WORKTREE.patch` records tracked, uncommitted differences at import time;
all source files, including untracked local HSC-G response files, are vendored.
`LOCAL_MODIFICATIONS.md` explains the scientific modifications. The packaging
additions (`pyproject.toml`, `MANIFEST.in`, `include_package_data`) make source
installation reproducible and include filter/SED assets in wheels.

Science fixtures were copied from the user's PHOTO-LENS03/032 research trees.
They contain PAUS and CFHT inputs and frozen PHOTO-LENS032 regions; see each
fixture's manifest for provenance and SHA-256 hashes. This notice does not grant
additional rights to third-party survey data or SED/filter datasets.
