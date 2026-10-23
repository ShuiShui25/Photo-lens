"""Reconstruct saved maximum-likelihood fits without a parameter search."""
from pathlib import Path
import autolens as al

def stage_tracer(output_dir: Path, stage: int) -> Path:
    matches = list(output_dir.glob(f"stage{stage}_*/**/files/tracer.json"))
    if stage == 4:
        current = [path for path in matches if "_clipsig1" in str(path)]
        if len(current) == 1:
            return current[0]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one stage-{stage} tracer under {output_dir}, found {len(matches)}"
        )
    return matches[0]


def maximum_likelihood_fit(config, bundle, tracer, stage: int):
    """Rebuild the deterministic fit represented by an on-disk tracer.

    Stage 4 requires the same stage-3 source adapt image and inversion settings
    used by the production likelihood.  No non-linear search is rerun.
    """
    if stage == 3:
        return al.FitImaging(dataset=bundle.dataset_masked, tracer=tracer)
    if stage != 4:
        raise ValueError(f"Exact plane export supports stages 3 and 4, not {stage}")

    stage3_path = stage_tracer(config.output_dir, 3)
    stage3_tracer = al.from_json(file_path=stage3_path)
    stage3_fit = al.FitImaging(
        dataset=bundle.dataset_masked, tracer=stage3_tracer
    )
    source_adapt_image = stage3_fit.model_images_of_planes_list[-1]
    pixel_sources = [
        galaxy for galaxy in tracer.galaxies if hasattr(galaxy, "pixelization")
    ]
    if len(pixel_sources) != 1:
        raise RuntimeError(
            f"Expected one stage-4 pixelized source, found {len(pixel_sources)}"
        )
    adapt_images = al.AdaptImages(
        galaxy_image_dict={pixel_sources[0]: source_adapt_image}
    )
    return al.FitImaging(
        dataset=bundle.dataset_masked,
        tracer=tracer,
        adapt_images=adapt_images,
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
