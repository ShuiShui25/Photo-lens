"""Joint posterior passing adapted from LD_mge joint8.

Common parameters retain their weighted empirical marginals and their Gaussian-
copula correlation. New parameters keep the ordinary AutoFit prior. This avoids
the scientifically unsafe independent-Gaussian projection between stages.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

_EPS = np.finfo(float).eps


def _flatten(path) -> tuple[str, ...]:
    parts: list[str] = []
    if isinstance(path, (tuple, list)):
        for item in path:
            parts.extend(_flatten(item))
    else:
        text = str(path).strip("'\"")
        if text:
            parts.append(text)
    return tuple(parts)


def path_labels(model) -> list[str]:
    return [
        ".".join(_flatten(list(model.all_paths_for_prior(prior))[0]))
        for prior in model.priors_ordered_by_id
    ]


def _marginal_grid(values: np.ndarray, weights: np.ndarray):
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return values, (cumulative - 0.5 * weights) / cumulative[-1]


def _normal_scores(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    positions = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    scores = np.empty(values.size)
    scores[order] = norm.ppf(np.clip(positions, _EPS, 1.0 - _EPS))
    return scores


def _nearest_pd_correlation(correlation: np.ndarray) -> np.ndarray:
    correlation = 0.5 * (correlation + correlation.T)
    value, vector = np.linalg.eigh(correlation)
    floor = max(float(np.max(value)) * 1.0e-8, 1.0e-10)
    repaired = (vector * np.maximum(value, floor)) @ vector.T
    scale = 1.0 / np.sqrt(np.maximum(np.diag(repaired), _EPS))
    repaired *= np.outer(scale, scale)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _copula_cholesky(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    scores = np.column_stack(
        [_normal_scores(values[:, index], weights) for index in range(values.shape[1])]
    )
    mean = np.average(scores, axis=0, weights=weights)
    centered = scores - mean
    covariance = (centered * weights[:, None]).T @ centered
    scale = np.sqrt(np.maximum(np.diag(covariance), _EPS))
    correlation = covariance / np.outer(scale, scale)
    correlation[~np.isfinite(correlation)] = 0.0
    correlation = np.clip(correlation, -0.999, 0.999)
    np.fill_diagonal(correlation, 1.0)
    return np.linalg.cholesky(_nearest_pd_correlation(correlation))


@dataclass
class EmpiricalPosteriorBlock:
    name: str
    target_indices: np.ndarray
    target_labels: tuple[str, ...]
    marginal_values: tuple[np.ndarray, ...]
    marginal_cdf: tuple[np.ndarray, ...]
    copula_cholesky: np.ndarray
    digest: str
    sample_count: int
    effective_sample_size: float

    def transform(self, unit: np.ndarray) -> np.ndarray:
        unit = np.clip(np.asarray(unit, dtype=float), _EPS, 1.0 - _EPS)
        correlated = norm.ppf(unit) @ self.copula_cholesky.T
        correlated_unit = np.clip(norm.cdf(correlated), _EPS, 1.0 - _EPS)
        output = np.empty_like(correlated_unit)
        for index in range(output.shape[1]):
            output[:, index] = np.interp(
                correlated_unit[:, index],
                self.marginal_cdf[index],
                self.marginal_values[index],
            )
        return output


def build_empirical_block(result, target_model, name: str) -> EmpiricalPosteriorBlock:
    source_labels = path_labels(result.model)
    target_labels = path_labels(target_model)
    source_by_label = {label: index for index, label in enumerate(source_labels)}
    pairs = [
        (target_index, source_by_label[label], label)
        for target_index, label in enumerate(target_labels)
        if label in source_by_label
    ]
    if not pairs:
        raise ValueError(f"No common nonlinear parameters for posterior block {name!r}")
    target_indices = np.asarray([item[0] for item in pairs], dtype=int)
    source_indices = np.asarray([item[1] for item in pairs], dtype=int)
    labels = tuple(item[2] for item in pairs)
    values = np.asarray(result.samples.parameter_lists, dtype=float)[:, source_indices]
    weights = np.asarray(result.samples.weight_list, dtype=float)
    valid = np.all(np.isfinite(values), axis=1) & np.isfinite(weights) & (weights > 0.0)
    values, weights = values[valid], weights[valid]
    if values.shape[0] < 2:
        raise ValueError("Upstream posterior has fewer than two valid weighted samples")
    weights /= weights.sum()
    grids = [_marginal_grid(values[:, i], weights) for i in range(values.shape[1])]
    cholesky = _copula_cholesky(values, weights)
    # AutoFit may regenerate and reorder auxiliary latent draws when a completed
    # result is restored. Those operations do not alter the upstream search,
    # but hashing the in-memory rows would create a new downstream search name.
    # The upstream AutoFit identifier already hashes its model/search result;
    # combine it with the mapped layout for a stable provenance digest.
    source_identifier = str(getattr(getattr(result, "paths", None), "identifier", ""))
    digest = hashlib.sha256()
    digest.update("joint8_gaussian_copula_v1".encode())
    digest.update(source_identifier.encode())
    digest.update("\n".join(labels).encode())
    block = EmpiricalPosteriorBlock(
        name=name,
        target_indices=target_indices,
        target_labels=labels,
        marginal_values=tuple(item[0] for item in grids),
        marginal_cdf=tuple(item[1] for item in grids),
        copula_cholesky=cholesky,
        digest=digest.hexdigest(),
        sample_count=int(values.shape[0]),
        effective_sample_size=float(1.0 / np.sum(weights**2)),
    )
    return block


def write_manifest(block: EmpiricalPosteriorBlock, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "method": "weighted_empirical_marginals_plus_gaussian_copula",
                "name": block.name,
                "digest": block.digest,
                "sample_count": block.sample_count,
                "effective_sample_size": block.effective_sample_size,
                "dimensions": len(block.target_indices),
                "parameters": list(block.target_labels),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class CorrelatedPriorVectorized:
    def __init__(self, model, blocks, base_cls):
        self.base = base_cls(model=model)
        self.blocks = tuple(blocks)

    def __call__(self, cube):
        cube_array = np.asarray(cube, dtype=float)
        one_dimensional = cube_array.ndim == 1
        cube_2d = cube_array[None, :] if one_dimensional else cube_array
        output = np.asarray(self.base(cube_2d), dtype=float)
        for block in self.blocks:
            output[:, block.target_indices] = block.transform(
                cube_2d[:, block.target_indices]
            )
        return output[0] if one_dimensional else output


@contextlib.contextmanager
def correlated_prior_transform(blocks):
    from autofit.non_linear.search.nest.nautilus import search as nautilus_search

    original = nautilus_search.PriorVectorized

    def factory(model):
        return CorrelatedPriorVectorized(model=model, blocks=blocks, base_cls=original)

    nautilus_search.PriorVectorized = factory
    try:
        yield
    finally:
        nautilus_search.PriorVectorized = original


def fit_with_correlated_posterior(search, model, analysis, block):
    with correlated_prior_transform([block]):
        return search.fit(model=model, analysis=analysis)
