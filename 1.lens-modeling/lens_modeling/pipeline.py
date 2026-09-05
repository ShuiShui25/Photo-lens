from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import autofit as af
import autolens as al
import autolens.plot as aplt

from .config import RunConfig
from .data import load_dataset, resolve_uncertainty_input
from .models import stage1_model, stage2_model, stage3_model, stage4_model
from .likelihood import (
    AnalysisImagingFigureOfMerit,
    LIKELIHOOD_SCHEMA,
    LIKELIHOOD_TAG,
    LIKELIHOOD_GUARD_TAG,
)
from .posterior import (
    build_empirical_block,
    fit_with_correlated_posterior,
    write_manifest,
)
from .products import write_wcs_products


# The v2.90 change is confined to derived-product export and therefore must
# not invalidate or silently rerun the already validated lens-model likelihood.
PIPELINE_REVISION = "photo_lens_unified_hsc_cfht_cbz_v1"
STAGE4_REGULARIZATION_REVISION = "clipsig1"


def _science_tag(config: RunConfig) -> str:
    payload = asdict(config)
    # Stage-4-only controls must not invalidate otherwise identical stage 1-3
    # results. They are encoded explicitly in the stage-4 search name instead.
    payload.pop("stage4_mesh_shape", None)
    payload.pop("stage4_use_jax", None)
    payload.pop("stage4_regularization", None)
    payload.pop("stage4_signal_scale", None)
    payload.pop("stage4_output_dir", None)
    payload = {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}
    # Explicitly invalidate AutoFit checkpoints when a shared numerical-model
    # choice changes without changing an input file or JSON configuration.
    payload["pipeline_revision"] = PIPELINE_REVISION
    uncertainty_kind, uncertainty_path, _ = resolve_uncertainty_input(config)
    payload["resolved_uncertainty_kind"] = uncertainty_kind
    payload["input_sha256"] = {
        key: hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in {
            "science_cutout": config.science_cutout,
            f"{uncertainty_kind}_cutout": uncertainty_path,
            "psf": config.psf,
            "positions_file": config.positions_file,
        }.items()
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:10]


def _search(config: RunConfig, stage: int, tag: str) -> af.Nautilus:
    return af.Nautilus(
        path_prefix=config.output_dir,
        name=f"stage{stage}_{tag}",
        n_live=config.n_live,
        n_eff=config.n_eff,
        n_batch=config.n_batch,
        number_of_cores=config.number_of_cores,
    )


def _analysis(bundle, config) -> al.AnalysisImaging:
    positions = None
    if bundle.image_positions:
        positions = [
            al.PositionsLH(
                positions=al.Grid2DIrregular(values=bundle.image_positions),
                threshold=config.positions_threshold,
            )
        ]
    return al.AnalysisImaging(
        dataset=bundle.dataset_masked,
        positions_likelihood_list=positions,
        use_jax=config.use_jax,
    )


def build_adapt_images_from_stage3(result_3, bundle) -> al.AdaptImages:
    """Stage-3's lensed-source model image, for signal-weighted regularization.

    ``adapt_data`` is an image-plane array (``mapper_util.adaptive_pixel_signals_from``
    indexes it by ``slim_index_for_sub_slim_index``), so the model image of the
    source galaxy from the previous stage is exactly what is wanted -- no extra
    fitting stage is needed to produce it.

    ``AdaptImages`` keys ``galaxy_name_image_dict`` by ``str(path_tuple)``, i.e.
    the literal ``"('galaxies', 'source')"``, not the ``"galaxies.source"`` the
    docstring suggests.

    ``load_latest_stage3_result`` returns a ``SimpleNamespace`` of samples rather
    than a full ``Result``, so the fit is rebuilt here from the stage-3 maximum
    likelihood instance.
    """
    tracer = al.Tracer(galaxies=result_3.samples.max_log_likelihood().galaxies)
    fit_3 = al.FitImaging(dataset=bundle.dataset_masked, tracer=tracer)
    source_image = fit_3.model_images_of_planes_list[-1]
    return al.AdaptImages(
        galaxy_name_image_dict={str(("galaxies", "source")): source_image}
    )


def _analysis_stage4(bundle, config, result_3=None) -> al.AnalysisImaging:
    positions = None
    if bundle.image_positions:
        positions = [
            al.PositionsLH(
                positions=al.Grid2DIrregular(values=bundle.image_positions),
                threshold=config.positions_threshold,
            )
        ]
    adapt_images = None
    if getattr(config, "stage4_regularization", "constant") != "constant":
        if result_3 is None:
            raise ValueError(
                "stage4_regularization requires an adapt image, so result_3 "
                "must be supplied to _analysis_stage4"
            )
        adapt_images = build_adapt_images_from_stage3(result_3, bundle)
    return AnalysisImagingFigureOfMerit(
        dataset=bundle.dataset_masked,
        positions_likelihood_list=positions,
        adapt_images=adapt_images,
        use_jax=config.stage4_use_jax,
        settings_inversion=al.SettingsInversion(
            use_positive_only_solver=True,
            positive_only_uses_p_initial=True,
            use_border_relocator=True,
            no_regularization_add_to_curvature_diag_value=1.0e-3,
            use_w_tilde_numpy=False,
            use_source_loop=False,
            tolerance=1.0e-8,
            maxiter=250,
        ),
    )


def load_latest_stage3_result(config: RunConfig):
    """Load, but never rerun, the completed stage-3 result for current inputs."""
    from autofit.aggregator.aggregator import Aggregator

    expected_tag = _science_tag(config)
    aggregator = Aggregator.from_directory(directory=config.output_dir)
    candidates = []
    for search, model, samples in zip(
        aggregator.values("search"),
        aggregator.values("model"),
        aggregator.values("samples"),
    ):
        if not search.name.startswith("stage3_double_source_"):
            continue
        if not search.name.endswith(f"_{expected_tag}"):
            continue
        result_path = Path(str(search.paths))
        completed = result_path / ".completed"
        archive = result_path.with_suffix(".zip")
        if not completed.is_file() and not archive.is_file():
            continue
        timestamp = max(
            path.stat().st_mtime for path in (completed, archive) if path.exists()
        )
        candidates.append(
            (
                timestamp,
                SimpleNamespace(
                    model=model,
                    samples=samples,
                    paths=search.paths,
                    search_name=search.name,
                ),
            )
        )
    if not candidates:
        raise FileNotFoundError(
            "No completed stage-3 result matches the current inputs and pipeline "
            f"revision (expected science tag {expected_tag}) in {config.output_dir}. "
            "Stage 4 refuses to rerun stages 1-3 implicitly."
        )
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def run_stage4(config: RunConfig):
    """Run only the fixed-mesh pixel-source stage from an on-disk stage 3."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(config)
    result_3 = load_latest_stage3_result(config)
    model_4 = stage4_model(config, result_3, bundle)
    block_34 = build_empirical_block(result_3, model_4, "stage3_to_stage4")
    required_prefixes = (
        "galaxies.lens.bulge.",
        "galaxies.lens.mass.",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(label.startswith(prefix) for label in block_34.target_labels)
    ]
    if missing:
        raise ValueError(
            "Stage 3 -> 4 posterior transfer is missing required lens blocks: "
            + ", ".join(missing)
        )
    # Stage 3 is always read from ``output_dir``; only stage 4 may be redirected.
    stage4_dir = Path(config.stage4_output_dir or config.output_dir)
    stage4_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = stage4_dir / "stage3_to_stage4_posterior.json"
    write_manifest(block_34, manifest_path)

    mesh_shape = tuple(int(value) for value in config.stage4_mesh_shape)
    common_tag = _science_tag(config)
    backend_tag = "jax" if config.stage4_use_jax else "cpu"
    scheme = getattr(config, "stage4_regularization", "constant")
    reg_tag = (
        "" if scheme == "constant"
        else f"_cbz_{STAGE4_REGULARIZATION_REVISION}"
    )
    search = af.Nautilus(
        path_prefix=stage4_dir,
        name=(
            f"stage4_pixel_source_{mesh_shape[0]}x{mesh_shape[1]}_"
            f"{backend_tag}_jp{block_34.digest[:10]}_{common_tag}_"
            f"{LIKELIHOOD_TAG}_{LIKELIHOOD_GUARD_TAG}_invld1{reg_tag}"
        ),
        n_live=config.n_live,
        n_eff=config.n_eff,
        n_batch=config.n_batch,
        number_of_cores=1 if config.stage4_use_jax else config.number_of_cores,
    )
    print(
        "Stage 4 only: "
        f"loaded stage 3={result_3.search_name} ({result_3.paths.identifier}); "
        f"fixed mesh={mesh_shape}; backend="
        f"{'jax' if config.stage4_use_jax else 'numpy_sparse_cpu'}; "
        f"posterior dimensions={len(block_34.target_indices)}"
    )
    analysis_4 = _analysis_stage4(bundle, config, result_3=result_3)
    result_4 = fit_with_correlated_posterior(
        search,
        model_4,
        analysis_4,
        block_34,
    )
    # Products are keyed off ``output_dir``, so point a copy of the config at the
    # stage-4 directory rather than threading a path through every writer.
    stage4_config = replace(config, output_dir=stage4_dir)
    preview = save_builtin_fit_subplot(result_4, stage4_config, 4)
    products = write_wcs_products(
        result_4, bundle, stage4_config, stage=4, output_name="wcs_products_stage4"
    )
    summary = {
        "system": config.system,
        "pipeline_revision": PIPELINE_REVISION,
        "stage4_regularization_revision": STAGE4_REGULARIZATION_REVISION,
        "stage": 4,
        "upstream_stage3": {
            "search_name": result_3.search_name,
            "identifier": str(result_3.paths.identifier),
        },
        "source": {
            "type": "pixelization",
            "mesh": "RectangularUniform",
            "shape": list(mesh_shape),
            "regularization": (
                "Constant" if scheme == "constant" else "ConstantPlusBrightnessZeroth"
            ),
            "regularization_prior": {
                "type": "LogUniformPrior",
                "lower_limit": 1.0e-3,
                "upper_limit": 1.0e4,
            },
            **(
                {}
                if scheme == "constant"
                else {
                    "coefficient_zeroth_prior": {
                        "type": "LogUniformPrior",
                        "lower_limit": 1.0e-1,
                        "upper_limit": 1.0e2,
                    },
                    "signal_scale": float(config.stage4_signal_scale),
                    "adapt_image": "stage3 galaxies.source model image (image plane)",
                }
            ),
        },
        "analysis_backend": "jax" if config.stage4_use_jax else "numpy_sparse_cpu",
        "jax_platform": (
            __import__("jax").default_backend() if config.stage4_use_jax else None
        ),
        "likelihood_schema": LIKELIHOOD_SCHEMA,
        "autofit_number_of_cores": (
            1 if config.stage4_use_jax else config.number_of_cores
        ),
        "nautilus_n_batch": config.n_batch,
        "settings_inversion": {
            "use_positive_only_solver": True,
            "positive_only_uses_p_initial": True,
            "use_border_relocator": True,
            "no_regularization_add_to_curvature_diag_value": 1.0e-3,
            "use_w_tilde_numpy": False,
            "use_source_loop": False,
            "tolerance": 1.0e-8,
            "maxiter": 250,
        },
        "posterior_transfer": {
            "method": "weighted_empirical_marginals_plus_gaussian_copula",
            "manifest": str(manifest_path),
            "dimensions": len(block_34.target_indices),
            "parameters": list(block_34.target_labels),
        },
        "log_evidence": float(result_4.samples.log_evidence),
        "reduced_chi_squared": float(
            result_4.max_log_likelihood_fit.reduced_chi_squared
        ),
        "fit_preview": str(preview),
        "wcs_products": {key: str(value) for key, value in products.items()},
    }
    summary_path = stage4_dir / "stage4_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result_4, summary_path


def save_builtin_fit_subplot(result, config: RunConfig, stage: int) -> Path:
    path = config.output_dir / "stage_previews"
    path.mkdir(parents=True, exist_ok=True)
    plotter = aplt.FitImagingPlotter(
        fit=result.max_log_likelihood_fit,
        mat_plot_2d=aplt.MatPlot2D(
            output=aplt.Output(
                path=str(path), filename=f"stage{stage}_fit", format="png"
            )
        ),
    )
    plotter.subplot_fit()
    return path / f"stage{stage}_fit.png"


def show_builtin_fit_subplots(results) -> None:
    for result in results:
        aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit).subplot_fit()


def run(config: RunConfig):
    config.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_dataset(config)
    print(
        "Model constraints: "
        f"mask centre={bundle.mask_centre_yx}, radius={bundle.mask_radius:.3f} arcsec; "
        f"oversampling centre={bundle.light_centre_yx if config.over_sample else None}; "
        f"image positions={list(bundle.image_positions)}; "
        f"PositionsLH threshold={config.positions_threshold:.3f} arcsec; "
        f"satellite positions={list(bundle.satellite_positions)}"
    )
    common_tag = _science_tag(config)

    model_1 = stage1_model(config, bundle)
    result_1 = _search(config, 1, f"sie_mge_{common_tag}").fit(
        model=model_1, analysis=_analysis(bundle, config)
    )
    save_builtin_fit_subplot(result_1, config, 1)

    model_2 = stage2_model(config, result_1, bundle)
    block_12 = build_empirical_block(result_1, model_2, "stage1_to_stage2")
    write_manifest(block_12, config.output_dir / "stage1_to_stage2_posterior.json")
    result_2 = fit_with_correlated_posterior(
        _search(config, 2, f"powerlaw_jp{block_12.digest[:10]}_{common_tag}"),
        model_2,
        _analysis(bundle, config),
        block_12,
    )
    save_builtin_fit_subplot(result_2, config, 2)

    model_3 = stage3_model(config, result_2, bundle)
    block_23 = build_empirical_block(result_2, model_3, "stage2_to_stage3")
    write_manifest(block_23, config.output_dir / "stage2_to_stage3_posterior.json")
    result_3 = fit_with_correlated_posterior(
        _search(config, 3, f"double_source_jp{block_23.digest[:10]}_{common_tag}"),
        model_3,
        _analysis(bundle, config),
        block_23,
    )
    save_builtin_fit_subplot(result_3, config, 3)
    products = write_wcs_products(result_3, bundle, config)

    summary = {
        "system": config.system,
        "pipeline_revision": PIPELINE_REVISION,
        "jax_requested": config.use_jax,
        "analysis_backend": "jax" if config.use_jax else "numpy_fnnls",
        "constraints": {
            "positions_file": str(config.positions_file),
            "mask_radius_arcsec": bundle.mask_radius,
            "mask_centre_yx_arcsec": list(bundle.mask_centre_yx),
            "oversampling_enabled": config.over_sample,
            "oversampling_centre_yx_arcsec": (
                list(bundle.light_centre_yx) if config.over_sample else None
            ),
            "image_positions_yx_arcsec": [list(item) for item in bundle.image_positions],
            "positions_threshold_arcsec": config.positions_threshold,
            "satellite_positions_yx_arcsec": [list(item) for item in bundle.satellite_positions],
        },
        "stages": {
            "1": {"mass": "Isothermal (SIE)", "source_mge": 10},
            "2": {"mass": "elliptical PowerLaw", "source_mge": 10},
            "3": {"mass": "elliptical PowerLaw", "source_mge": "2x10 independent-centre"},
        },
        "log_evidence": {
            "stage1": float(result_1.samples.log_evidence),
            "stage2": float(result_2.samples.log_evidence),
            "stage3": float(result_3.samples.log_evidence),
        },
        "reduced_chi_squared": {
            "stage1": float(result_1.max_log_likelihood_fit.reduced_chi_squared),
            "stage2": float(result_2.max_log_likelihood_fit.reduced_chi_squared),
            "stage3": float(result_3.max_log_likelihood_fit.reduced_chi_squared),
        },
        "wcs_products": {key: str(value) for key, value in products.items()},
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return (result_1, result_2, result_3), products
