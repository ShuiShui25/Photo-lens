from __future__ import annotations

import numpy as np

import autofit as af
import autolens as al

from .regularization import ConstantPlusBrightnessZeroth


def mge_basis(
    *,
    max_sigma: float,
    gaussians_per_basis: int,
    total_bases: int,
    centre_yx: tuple[float, float] | None = None,
    centre_sigma: float = 0.3,
    shared_centre=None,
):
    """Build elliptical linear-Gaussian bases with fixed log-spaced scales.

    Each basis has its own ellipticity while all profiles returned by one call
    share a center. Intensities remain linear parameters solved by AutoLens.
    """
    if shared_centre is None:
        mean_y, mean_x = centre_yx or (0.0, 0.0)
        centre_0 = af.GaussianPrior(mean=mean_y, sigma=centre_sigma)
        centre_1 = af.GaussianPrior(mean=mean_x, sigma=centre_sigma)
    else:
        centre_0 = shared_centre.centre_0
        centre_1 = shared_centre.centre_1

    sigmas = np.logspace(-2.0, np.log10(max_sigma), gaussians_per_basis)
    profiles = []
    for _ in range(total_bases):
        group = af.Collection(
            af.Model(al.lp_linear.Gaussian) for _ in range(gaussians_per_basis)
        )
        group[0].ell_comps.ell_comps_0 = af.UniformPrior(-0.7, 0.7)
        group[0].ell_comps.ell_comps_1 = af.UniformPrior(-0.7, 0.7)
        for index, gaussian in enumerate(group):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            gaussian.ell_comps = group[0].ell_comps
            gaussian.sigma = float(sigmas[index])
            profiles.append(gaussian)
    return af.Model(al.lp_basis.Basis, profile_list=profiles)


def stage1_model(config, bundle):
    centre_yx = bundle.light_centre_yx
    lens_light = mge_basis(
        max_sigma=bundle.mask_radius,
        gaussians_per_basis=config.lens_gaussians_per_basis,
        total_bases=config.lens_bases,
        centre_yx=centre_yx,
        centre_sigma=config.light_centre_sigma,
    )
    mass = af.Model(al.mp.Isothermal)
    mass.centre.centre_0 = af.GaussianPrior(
        mean=bundle.mass_centre_yx[0], sigma=config.mass_centre_sigma
    )
    mass.centre.centre_1 = af.GaussianPrior(
        mean=bundle.mass_centre_yx[1], sigma=config.mass_centre_sigma
    )
    mass.einstein_radius = af.GaussianPrior(
        mean=config.einstein_radius_mean, sigma=0.3
    )
    source_light = mge_basis(
        max_sigma=1.5,
        gaussians_per_basis=config.source_gaussians,
        total_bases=1,
        centre_yx=(0.0, 0.0),
        centre_sigma=0.3,
    )
    lens = af.Model(
        al.Galaxy,
        redshift=config.z_lens,
        bulge=lens_light,
        mass=mass,
        shear=af.Model(al.mp.ExternalShear),
    )
    source = af.Model(al.Galaxy, redshift=config.z_source, bulge=source_light)
    satellites = {}
    for index, satellite_centre in enumerate(bundle.satellite_positions):
        satellite_light = mge_basis(
            max_sigma=bundle.mask_radius,
            gaussians_per_basis=config.lens_gaussians_per_basis,
            total_bases=config.lens_bases,
            centre_yx=satellite_centre,
            centre_sigma=config.satellite_light_centre_sigma,
        )
        satellites[f"satellite_{index}"] = af.Model(
            al.Galaxy, redshift=config.z_lens, bulge=satellite_light
        )
    return af.Collection(
        galaxies=af.Collection(lens=lens, **satellites, source=source)
    )


def stage2_model(config, result_1, bundle):
    previous = result_1.model
    mass = af.Model(al.mp.PowerLaw)
    mass.centre = previous.galaxies.lens.mass.centre
    mass.ell_comps = previous.galaxies.lens.mass.ell_comps
    mass.einstein_radius = previous.galaxies.lens.mass.einstein_radius
    mass.slope = af.UniformPrior(lower_limit=1.5, upper_limit=3.0)
    lens = af.Model(
        al.Galaxy,
        redshift=config.z_lens,
        bulge=previous.galaxies.lens.bulge,
        mass=mass,
        shear=previous.galaxies.lens.shear,
    )
    source = af.Model(
        al.Galaxy,
        redshift=config.z_source,
        bulge=previous.galaxies.source.bulge,
    )
    satellites = {
        f"satellite_{index}": getattr(previous.galaxies, f"satellite_{index}")
        for index in range(len(bundle.satellite_positions))
    }
    return af.Collection(
        galaxies=af.Collection(lens=lens, **satellites, source=source)
    )


def stage3_model(config, result_2, bundle):
    previous = result_2.model
    first_source = previous.galaxies.source.bulge
    second_source = mge_basis(
        max_sigma=1.5,
        gaussians_per_basis=config.source_gaussians,
        total_bases=1,
        centre_yx=(0.0, 0.0),
        centre_sigma=0.5,
    )
    lens = af.Model(
        al.Galaxy,
        redshift=config.z_lens,
        bulge=previous.galaxies.lens.bulge,
        mass=previous.galaxies.lens.mass,
        shear=previous.galaxies.lens.shear,
    )
    source = af.Model(
        al.Galaxy,
        redshift=config.z_source,
        bulge=first_source,
        disk=second_source,
    )
    satellites = {
        f"satellite_{index}": getattr(previous.galaxies, f"satellite_{index}")
        for index in range(len(bundle.satellite_positions))
    }
    return af.Collection(
        galaxies=af.Collection(lens=lens, **satellites, source=source)
    )


def stage4_model(config, result_3, bundle):
    """Stage 4: retain the lens posterior and replace the source by fixed pixels."""
    previous = result_3.model
    lens = af.Model(
        al.Galaxy,
        redshift=config.z_lens,
        bulge=previous.galaxies.lens.bulge,
        mass=previous.galaxies.lens.mass,
        shear=previous.galaxies.lens.shear,
    )
    mesh_shape = tuple(int(value) for value in config.stage4_mesh_shape)
    if len(mesh_shape) != 2 or any(value <= 1 for value in mesh_shape):
        raise ValueError(f"stage4_mesh_shape must contain two integers > 1: {mesh_shape}")
    scheme = getattr(config, "stage4_regularization", "constant")
    if scheme == "constant":
        regularization = af.Model(al.reg.Constant)
    elif scheme == "constant_brightness_zeroth":
        # Constant regularization penalises only differences between neighbouring
        # pixels, so a smooth pedestal across the source plane is nearly free and
        # the weakly-constrained outskirts absorb flux that is degenerate with the
        # lens MGE.  The signal-weighted zeroth-order term prices that pedestal
        # while leaving the bright source pixels alone.
        regularization = af.Model(
            ConstantPlusBrightnessZeroth,
            signal_scale=float(getattr(config, "stage4_signal_scale", 1.0)),
        )
        regularization.coefficient_zeroth = af.LogUniformPrior(
            lower_limit=1.0e-1,
            upper_limit=1.0e2,
        )
    else:
        raise ValueError(f"unknown stage4_regularization: {scheme!r}")

    pixelization = af.Model(
        al.Pixelization,
        mesh=af.Model(al.mesh.RectangularUniform, shape=mesh_shape),
        regularization=regularization,
    )
    pixelization.regularization.coefficient = af.LogUniformPrior(
        lower_limit=1.0e-3,
        upper_limit=1.0e4,
    )
    source = af.Model(
        al.Galaxy,
        redshift=config.z_source,
        pixelization=pixelization,
    )
    satellites = {
        f"satellite_{index}": getattr(previous.galaxies, f"satellite_{index}")
        for index in range(len(bundle.satellite_positions))
    }
    return af.Collection(
        galaxies=af.Collection(lens=lens, **satellites, source=source)
    )
