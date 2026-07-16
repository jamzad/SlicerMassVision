from __future__ import annotations

"""
MassVision / EmbedVision spatial-transcriptomics converter
===========================================================

Purpose
-------
Read a spatial-transcriptomics ``.h5ad`` file, preprocess the spot-by-gene
matrix using standard AnnData/Scanpy workflows, and create a regular
non-interpolated spot-footprint raster compatible with image/cube-based
MassVision workflows.

The scientific source of truth remains one vector per measured spot::

    spot_matrix.shape == (n_spots, n_selected_genes)

The raster cube is a visualization and software-compatibility representation::

    raster_cube.shape == (height, width, n_selected_genes)

Every raster pixel inside a spot's circular footprint receives the SAME vector
as that spot. Pixels between spots remain unmeasured (NaN by default). No gene
values are interpolated between spots.

Scientific interpretation
-------------------------
* AnnData is used to read and organize the ``.h5ad`` data.
* Scanpy is used for the core transcriptomics preprocessing:
  QC, filtering, total-count normalization, log1p, and highly variable genes.
* Squidpy is OPTIONAL. When enabled, it uses a spatial-neighbor graph and
  Moran's I to retain genes with non-random spatial structure.
* Squidpy/Moran's I is NOT primarily a method for removing correlated or
  redundant genes. It is an optional spatial-informativeness filter.
* The circular raster is equivalent to drawing conventional colored Visium
  spots, except that circles are rasterized into pixels for MassVision.
* Downstream statistical analyses should use ``spot_matrix`` or
  ``processed_adata.X`` rather than treating duplicate pixels within a spot as
  independent measurements.

Expected common AnnData structure
---------------------------------
The converter supports common conventions but ``.h5ad`` is flexible, so users
should inspect new datasets before conversion.

Typical fields::

    adata.X                       # main spots x genes matrix
    adata.layers["counts"]        # raw counts, when retained
    adata.raw.X                   # sometimes raw or less-processed values
    adata.obs                     # spot metadata
    adata.var                     # gene metadata
    adata.obsm["spatial"]         # spot center coordinates, usually x/y
    adata.uns["spatial"]          # optional Visium images and scalefactors

For standard Visium objects created by Scanpy/Space Ranger workflows,
``obsm['spatial']`` usually contains full-resolution pathology-image pixel
coordinates and the physical display diameter is commonly stored at::

    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ]

Dependencies
------------
Required::

    pip install anndata scanpy numpy pandas scipy

The default Scanpy ``seurat_v3`` HVG method can require ``scikit-misc``::

    pip install scikit-misc

Optional spatial-gene ranking::

    pip install squidpy

Squidpy is imported optionally. The main conversion works without it when
``spatial_gene_selection`` is disabled, which is the default.

Quick start
-----------

1. Standard raw-count Visium ``.h5ad``::

    result = h5ad_to_spot_footprint_cube("sample.h5ad")

    print(result.spot_matrix.shape)
    print(result.raster_cube.shape)
    print(result.gene_names[:10])
    print(result.metadata)

2. Explicit count layer and 1,500 genes::

    result = h5ad_to_spot_footprint_cube(
        "sample.h5ad",
        processing={
            "matrix_source": "layer:counts",
            "data_kind": "counts",
            "n_top_genes": 1500,
            "hvg_flavor": "seurat_v3",
        },
        raster={
            "target_min_dimension": 256,
            "spot_size_factor": 1.0,
        },
    )

3. Already normalized/log-transformed ``adata.X``::

    result = h5ad_to_spot_footprint_cube(
        "processed_sample.h5ad",
        processing={
            "matrix_source": "X",
            "data_kind": "processed",
            "normalize_total": False,
            "log1p": False,
            "gene_selection": "variance",
            "n_top_genes": 1000,
        },
    )

4. Keep specific genes even when they are not selected automatically::

    result = h5ad_to_spot_footprint_cube(
        "sample.h5ad",
        processing={
            "include_genes": ["EPCAM", "KRT8", "KRT18", "COL1A1"],
        },
    )

5. Optional Squidpy spatial-gene ranking::

    result = h5ad_to_spot_footprint_cube(
        "sample.h5ad",
        processing={
            "gene_selection": "hvg_then_spatial",
            "spatial_gene_selection": True,
            "spatial_candidate_genes": 2000,
            "n_top_genes": 1000,
            "moran_min_i": None,
            "moran_max_fdr": None,
        },
    )

   This first identifies candidate HVGs with Scanpy, then ranks those candidates
   by Moran's I. It is more computationally expensive and is not required for a
   standard visualization cube.

6. Multiple libraries/samples stored in one ``.h5ad``::

    result = h5ad_to_spot_footprint_cube(
        "combined.h5ad",
        processing={
            "library_id": "BC_1_515",
            "obs_library_key": "library_id",
        },
    )

7. Avoid allocating a large dense cube and render channels lazily::

    result = h5ad_to_spot_footprint_cube(
        "sample.h5ad",
        raster={"materialize_cube": False},
    )

    epcam_index = list(result.gene_names).index("EPCAM")
    epcam_image = render_spot_channels(
        result.spot_matrix,
        result.spot_id_map,
        epcam_index,
    )

8. Convert a MassVision ROI mask back to unique measured spots::

    selected_ids = spot_ids_from_roi(
        result.spot_id_map,
        roi_mask,
        selection="intersects",
    )
    selected_expression = result.spot_matrix[selected_ids]

Main output fields
------------------
``result.processed_adata``
    AnnData containing retained spots and selected genes. ``X`` contains the
    values used for visualization. Raw counts are preserved in
    ``layers['massvision_counts']`` when count data are available.

``result.spot_matrix``
    Dense float array of shape ``n_spots x n_selected_genes``.

``result.raster_cube``
    Dense or memory-mapped array of shape ``H x W x n_selected_genes``. It can
    be ``None`` when lazy rendering is selected.

``result.spot_id_map``
    Integer image of shape ``H x W``. Each measured footprint contains the
    corresponding row index in ``spot_matrix``. Background is ``-1``.

``result.valid_mask``
    Boolean image that is True only inside measured spot footprints.

``result.gene_table``
    Gene names, IDs, selection information, HVG values, and optional Moran's I.

Recommended default for MassVision
----------------------------------
For a first visualization-oriented implementation, use:

* raw counts when available;
* conservative spot/gene filtering;
* total-count normalization to 10,000;
* natural-log ``log1p`` transformation;
* 1,000 Scanpy HVGs;
* no gene-wise z-scoring;
* true spot diameter with ``spot_size_factor=1.0``;
* NaN background plus ``valid_mask``;
* no spatial interpolation;
* Squidpy disabled unless the user explicitly requests spatial gene selection.

Coordinate and spot-size warning
--------------------------------
AnnData does not enforce coordinate units. If ``spot_diameter_fullres`` is used,
coordinates must be in the matching full-resolution image-pixel coordinate
system. If the dataset uses microns, low-resolution image coordinates, or a
custom transformed coordinate system, provide ``raster['spot_diameter']`` in
those same coordinate units.
"""

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree

# Squidpy is deliberately optional. Importing it through try/except allows the
# normal AnnData + Scanpy workflow to run in environments where Squidpy is not
# installed. Spatial gene selection raises a clear error only when requested.
try:
    import squidpy as sq
except ImportError:  # pragma: no cover - depends on the user's environment
    sq = None


@dataclass
class SpotFootprintRasterResult:
    """Result returned by :func:`h5ad_to_spot_footprint_cube`."""

    processed_adata: ad.AnnData
    """Retained spots and selected genes; ``X`` contains display values."""

    spot_matrix: np.ndarray
    """Processed spot-by-gene values, shape ``(n_spots, n_genes)``."""

    gene_names: np.ndarray
    """Unique display gene names in the same order as ``spot_matrix`` columns."""

    gene_ids: np.ndarray
    """Stable gene identifiers when available."""

    gene_table: pd.DataFrame
    """Gene metadata, selection fields, and optional spatial statistics."""

    barcodes: np.ndarray
    """Retained spot barcodes/AnnData observation names."""

    spot_table: pd.DataFrame
    """Retained observation metadata plus source and raster coordinates."""

    raster_cube: np.ndarray | np.memmap | None
    """Raster cube ``(height, width, n_genes)`` or ``None`` in lazy mode."""

    spot_id_map: np.ndarray
    """Integer label map; ``-1`` means unmeasured background."""

    valid_mask: np.ndarray
    """Boolean mask identifying measured circular spot footprints."""

    raster_spot_centers_xy: np.ndarray
    """Spot centers in output-raster x/y coordinates."""

    metadata: dict[str, Any]
    """Resolved settings, transformations, warnings, and diagnostics."""


# ---------------------------------------------------------------------------
# User-facing configuration
# ---------------------------------------------------------------------------

DEFAULT_PROCESSING: dict[str, Any] = {
    # Matrix input ---------------------------------------------------------
    # "auto" preference order:
    #   layers['counts'] -> layers['raw_counts'] -> count-like raw.X -> X
    # Explicit examples: "X", "raw", "layer:counts", "layer:lognorm".
    "matrix_source": "auto",

    # "auto" checks whether sampled non-zero values are non-negative integers.
    # Use an explicit value for important datasets.
    "data_kind": "auto",  # auto, counts, processed

    # Spatial/sample selection -------------------------------------------
    "spatial_key": "spatial",  # adata.obsm key
    "library_id": None,  # uns['spatial'] library; inferred if only one exists
    "obs_library_key": None,  # e.g. 'library_id' for concatenated objects
    "tissue_only": True,  # uses obs['in_tissue'] when present

    # Spot quality control ------------------------------------------------
    # Defaults are intentionally conservative because appropriate thresholds
    # vary substantially by tissue, chemistry, sequencing depth, and study.
    "min_counts_per_spot": 0,
    "min_genes_per_spot": 0,
    "max_counts_per_spot": None,
    "max_genes_per_spot": None,
    "max_pct_mito": None,

    # Gene quality control ------------------------------------------------
    "min_spots_per_gene": 3,
    "min_total_counts_per_gene": 0,

    # Mitochondrial/ribosomal annotations are useful QC fields. They are not
    # removed by default because exclusion is study-dependent.
    "mitochondrial_prefixes": ["MT-"],
    "ribosomal_prefixes": ["RPL", "RPS"],
    "exclude_mitochondrial_genes": False,
    "exclude_ribosomal_genes": False,
    "exclude_gene_prefixes": [],
    "exclude_gene_regex": None,

    # Scanpy transformation ----------------------------------------------
    # In auto behavior, raw counts are normalized/logged; processed data are
    # left unchanged. Set explicit booleans to override.
    "normalize_total": "auto",  # auto, True, False
    "target_sum": 10_000.0,
    "exclude_highly_expressed_from_normalization": False,
    "normalization_max_fraction": 0.05,
    "log1p": "auto",  # auto, True, False

    # Gene selection ------------------------------------------------------
    # Recommended default: Scanpy highly variable genes.
    # Options:
    #   "hvg"                Scanpy HVGs
    #   "hvg_then_spatial"   HVG candidates, then Squidpy Moran's I ranking
    #   "variance"           top variance on the final display values
    #   "all"                retain all genes surviving QC
    #   "gene_list"          retain only include_genes
    "gene_selection": "hvg",
    "n_top_genes": 1_000,

    # seurat_v3 uses raw counts and commonly requires scikit-misc.
    # If it is unavailable and allow_hvg_fallback=True, the code falls back to
    # classic "seurat" on normalized/logged values with a warning.
    "hvg_flavor": "seurat_v3",
    "hvg_n_bins": 20,
    "allow_hvg_fallback": True,

    # User-requested genes are retained even if they are not selected by HVG
    # or spatial ranking. Matching checks var_names, gene symbols, and IDs.
    "include_genes": [],
    "error_on_missing_include_genes": True,

    # Optional Squidpy spatial selection ---------------------------------
    # Disabled by default. It is useful when a user wants genes with clear
    # spatial organization, not merely high variance.
    "spatial_gene_selection": False,
    "spatial_candidate_genes": 2_000,
    "spatial_coord_type": "grid",  # 'grid' for Visium; 'generic' otherwise
    "spatial_n_neighs": 6,
    "spatial_n_rings": 1,
    "spatial_radius": None,  # only for generic coordinate graphs
    "spatial_delaunay": False,
    "moran_min_i": None,
    "moran_max_fdr": None,
    "moran_n_perms": None,
    "moran_n_jobs": 1,
    "random_seed": 0,

    # Output matrix -------------------------------------------------------
    # Gene-wise z-scoring is OFF because it changes expression units and
    # introduces negative values. It can be useful for some feature-space
    # visualizations but should be a deliberate user choice.
    "scale_genes": False,
    "scale_zero_center": True,
    "scale_max_value": 10.0,
    "dtype": "float32",
}


DEFAULT_RASTER: dict[str, Any] = {
    # Output resolution ---------------------------------------------------
    # min_dimension: set the smaller output dimension approximately to target.
    # coordinate_pixel_size: one output pixel equals the specified number of
    # source-coordinate units.
    "resolution_mode": "min_dimension",
    "target_min_dimension": 256,
    "coordinate_units_per_raster_pixel": None,

    # Spot geometry -------------------------------------------------------
    # If None, attempt to read spot_diameter_fullres. Final fallback estimates
    # diameter from nearest-neighbor spacing using the 55/100 Visium ratio.
    "spot_diameter": None,
    "allow_diameter_estimation": True,
    "spot_diameter_to_spacing_ratio": 0.55,

    # 1.0 corresponds to the inferred/physical capture footprint. Larger or
    # smaller values are conventional display scaling, not interpolation.
    "spot_size_factor": 1.0,

    # Preserve true scaled size by default. Enable only for readability at very
    # low raster resolutions; doing so changes the displayed footprint size.
    "enforce_min_spot_diameter": False,
    "min_spot_diameter_px": 3.0,

    # Canvas --------------------------------------------------------------
    "crop_to_spots": True,
    "margin_spot_diameters": 0.5,
    "background_value": np.nan,

    # Dense cube allocation ----------------------------------------------
    "materialize_cube": True,
    "max_cube_gb": 2.0,
    "memmap_path": None,
}


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _merge_config(
    defaults: Mapping[str, Any],
    updates: Mapping[str, Any] | None,
    *,
    config_name: str,
) -> dict[str, Any]:
    """Merge a user dictionary with defaults and reject misspelled keys."""
    merged = dict(defaults)
    if updates:
        unknown = set(updates) - set(defaults)
        if unknown:
            raise KeyError(
                f"Unknown {config_name} configuration key(s): {sorted(unknown)}"
            )
        merged.update(updates)
    return merged


def _as_csr(matrix: Any) -> sparse.csr_matrix:
    """Return an independent CSR representation of a 2D matrix."""
    if sparse.issparse(matrix):
        return matrix.tocsr(copy=True)
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"Expression matrix must be 2D; got shape {array.shape}")
    return sparse.csr_matrix(array)


def _sample_nonzero_values(
    matrix: sparse.csr_matrix,
    max_values: int = 50_000,
) -> np.ndarray:
    data = np.asarray(matrix.data, dtype=np.float64)
    if data.size <= max_values:
        return data
    rng = np.random.default_rng(0)
    indices = rng.choice(data.size, size=max_values, replace=False)
    return data[indices]


def _looks_like_counts(matrix: sparse.csr_matrix) -> bool:
    """Heuristic count check used only for ``data_kind='auto'``."""
    values = _sample_nonzero_values(matrix)
    if values.size == 0:
        return True
    if not np.all(np.isfinite(values)) or np.min(values) < 0:
        return False
    integer_fraction = np.mean(np.isclose(values, np.round(values), atol=1e-6))
    return bool(integer_fraction >= 0.995)


def _make_unique(names: Sequence[str]) -> np.ndarray:
    """Make display names unique without changing their original order."""
    counts: dict[str, int] = {}
    output: list[str] = []
    for raw_name in names:
        name = str(raw_name)
        counts[name] = counts.get(name, 0) + 1
        output.append(name if counts[name] == 1 else f"{name}-{counts[name]}")
    return np.asarray(output, dtype=object)


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def _truthy_tissue_mask(values: pd.Series) -> np.ndarray:
    """Interpret common integer, boolean, and string in_tissue encodings."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).to_numpy() > 0
    strings = values.astype(str).str.strip().str.lower()
    return strings.isin({"true", "t", "yes", "y", "1", "in_tissue"}).to_numpy()


# ---------------------------------------------------------------------------
# AnnData input inspection and matrix resolution
# ---------------------------------------------------------------------------


def inspect_h5ad(h5ad_path: str | Path) -> dict[str, Any]:
    """Return a compact structural summary before choosing conversion options.

    This helper intentionally does not modify or preprocess the file.
    """
    path = Path(h5ad_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    adata = ad.read_h5ad(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "shape": tuple(adata.shape),
        "layers": list(adata.layers.keys()),
        "obsm_keys": list(adata.obsm.keys()),
        "obsp_keys": list(adata.obsp.keys()),
        "uns_keys": list(adata.uns.keys()),
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "has_raw": adata.raw is not None,
        "x_looks_like_counts": _looks_like_counts(_as_csr(adata.X)),
    }

    spatial = adata.uns.get("spatial")
    summary["spatial_library_ids"] = (
        list(spatial.keys()) if isinstance(spatial, Mapping) else []
    )
    return summary


def _resolve_library_metadata(
    adata: ad.AnnData,
    requested_library_id: str | None,
) -> tuple[str | None, Mapping[str, Any] | None]:
    spatial_uns = adata.uns.get("spatial")
    if not isinstance(spatial_uns, Mapping) or len(spatial_uns) == 0:
        return requested_library_id, None

    available = [str(key) for key in spatial_uns.keys()]
    library_id = requested_library_id
    if library_id is None:
        if len(available) == 1:
            library_id = available[0]
        else:
            raise ValueError(
                "This h5ad contains multiple entries in adata.uns['spatial']. "
                f"Set processing['library_id']. Available values: {available}"
            )

    if library_id not in spatial_uns:
        raise KeyError(
            f"library_id={library_id!r} is not in adata.uns['spatial']; "
            f"available values: {available}"
        )

    entry = spatial_uns[library_id]
    return library_id, entry if isinstance(entry, Mapping) else None


def _resolve_expression_source(
    adata: ad.AnnData,
    source: str,
) -> tuple[sparse.csr_matrix, pd.DataFrame, str]:
    """Resolve a count/processed matrix and its matching gene metadata."""
    requested = str(source)

    if requested == "auto":
        if "counts" in adata.layers:
            requested = "layer:counts"
        elif "raw_counts" in adata.layers:
            requested = "layer:raw_counts"
        elif adata.raw is not None and _looks_like_counts(_as_csr(adata.raw.X)):
            requested = "raw"
        else:
            requested = "X"

    if requested == "X":
        matrix = adata.X
        var = adata.var.copy()
    elif requested == "raw":
        if adata.raw is None:
            raise KeyError("matrix_source='raw' requested, but adata.raw is absent")
        matrix = adata.raw.X
        var = adata.raw.var.copy()
    elif requested.startswith("layer:"):
        layer_name = requested.split(":", 1)[1]
        if layer_name not in adata.layers:
            raise KeyError(
                f"AnnData layer {layer_name!r} was not found. "
                f"Available layers: {list(adata.layers.keys())}"
            )
        matrix = adata.layers[layer_name]
        var = adata.var.copy()
    else:
        raise ValueError(
            "matrix_source must be 'auto', 'X', 'raw', or 'layer:<name>'"
        )

    resolved = _as_csr(matrix)
    if resolved.shape[0] != adata.n_obs:
        raise ValueError("Resolved matrix observation count does not match adata.obs")
    if resolved.shape[1] != len(var):
        raise ValueError("Resolved matrix gene count does not match its var table")
    return resolved, var, requested


def _build_gene_table(var: pd.DataFrame) -> pd.DataFrame:
    """Create stable gene display names and identifiers from common columns."""
    original_var_names = np.asarray(var.index.astype(str), dtype=object)

    symbol_column = _first_existing_column(
        var,
        (
            "gene_symbols",
            "gene_symbol",
            "symbol",
            "feature_name",
            "gene_name",
        ),
    )
    id_column = _first_existing_column(
        var,
        ("gene_ids", "gene_id", "feature_id", "ensembl_id", "id"),
    )

    symbols = (
        var[symbol_column].astype(str).to_numpy(dtype=object)
        if symbol_column is not None
        else original_var_names.copy()
    )
    gene_ids = (
        var[id_column].astype(str).to_numpy(dtype=object)
        if id_column is not None
        else original_var_names.copy()
    )

    return pd.DataFrame(
        {
            "original_var_name": original_var_names,
            "gene_symbol": symbols,
            "gene_name": _make_unique(symbols),
            "gene_id": gene_ids,
        }
    )


def _match_requested_genes(
    gene_table: pd.DataFrame,
    requested_genes: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Match user genes against symbols, IDs, and original var names."""
    requested = {str(value).upper() for value in requested_genes}
    if not requested:
        return np.zeros(len(gene_table), dtype=bool), []

    matched_mask = np.zeros(len(gene_table), dtype=bool)
    found: set[str] = set()
    for column in ("gene_name", "gene_symbol", "gene_id", "original_var_name"):
        values = gene_table[column].astype(str).str.upper().to_numpy(dtype=object)
        current = np.isin(values, list(requested))
        matched_mask |= current
        found.update(set(values[current]).intersection(requested))

    return matched_mask, sorted(requested - found)


# ---------------------------------------------------------------------------
# Scanpy preprocessing and optional Squidpy selection
# ---------------------------------------------------------------------------


def _annotate_gene_groups(adata: ad.AnnData, config: Mapping[str, Any]) -> None:
    symbols = adata.var["gene_symbol"].astype(str).str.upper()
    mito_prefixes = tuple(str(x).upper() for x in config["mitochondrial_prefixes"])
    ribo_prefixes = tuple(str(x).upper() for x in config["ribosomal_prefixes"])

    adata.var["mt"] = (
        symbols.str.startswith(mito_prefixes).to_numpy()
        if mito_prefixes
        else False
    )
    adata.var["ribo"] = (
        symbols.str.startswith(ribo_prefixes).to_numpy()
        if ribo_prefixes
        else False
    )


def _filter_spots_with_qc(adata: ad.AnnData, config: Mapping[str, Any]) -> ad.AnnData:
    """Calculate Scanpy QC metrics and apply configured spot thresholds."""
    qc_vars = [name for name in ("mt", "ribo") if name in adata.var]
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=qc_vars,
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    keep = np.ones(adata.n_obs, dtype=bool)

    minimum = int(config["min_counts_per_spot"])
    if minimum > 0:
        keep &= adata.obs["total_counts"].to_numpy() >= minimum

    minimum = int(config["min_genes_per_spot"])
    if minimum > 0:
        keep &= adata.obs["n_genes_by_counts"].to_numpy() >= minimum

    maximum = config["max_counts_per_spot"]
    if maximum is not None:
        keep &= adata.obs["total_counts"].to_numpy() <= float(maximum)

    maximum = config["max_genes_per_spot"]
    if maximum is not None:
        keep &= adata.obs["n_genes_by_counts"].to_numpy() <= int(maximum)

    maximum = config["max_pct_mito"]
    if maximum is not None:
        if "pct_counts_mt" not in adata.obs:
            raise ValueError("Mitochondrial QC was requested but no mt genes were identified")
        keep &= adata.obs["pct_counts_mt"].to_numpy() <= float(maximum)

    if not np.any(keep):
        raise ValueError("All spots were removed by tissue/QC filtering")
    return adata[keep].copy()


def _filter_genes_with_scanpy(
    adata: ad.AnnData,
    config: Mapping[str, Any],
) -> ad.AnnData:
    """Apply minimum detection/count filters and optional named exclusions."""
    keep = np.ones(adata.n_vars, dtype=bool)

    min_spots = int(config["min_spots_per_gene"])
    if min_spots > 0:
        detected = np.asarray((adata.X != 0).sum(axis=0)).ravel()
        keep &= detected >= min_spots

    min_counts = float(config["min_total_counts_per_gene"])
    if min_counts > 0:
        totals = np.asarray(adata.X.sum(axis=0)).ravel()
        keep &= totals >= min_counts

    if bool(config["exclude_mitochondrial_genes"]):
        keep &= ~adata.var["mt"].to_numpy(dtype=bool)
    if bool(config["exclude_ribosomal_genes"]):
        keep &= ~adata.var["ribo"].to_numpy(dtype=bool)

    prefixes = tuple(str(x).upper() for x in config["exclude_gene_prefixes"])
    if prefixes:
        symbols = adata.var["gene_symbol"].astype(str).str.upper()
        keep &= ~symbols.str.startswith(prefixes).to_numpy()

    regex = config["exclude_gene_regex"]
    if regex:
        excluded = adata.var["gene_symbol"].astype(str).str.contains(
            re.compile(str(regex)),
            regex=True,
            na=False,
        )
        keep &= ~excluded.to_numpy()

    if not np.any(keep):
        raise ValueError("All genes were removed by gene QC/exclusion settings")
    return adata[:, keep].copy()


def _resolve_auto_boolean(value: Any, *, data_kind: str) -> bool:
    if value == "auto":
        return data_kind == "counts"
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError("Expected 'auto', True, or False")


def _normalize_with_scanpy(
    adata: ad.AnnData,
    config: Mapping[str, Any],
    *,
    data_kind: str,
) -> tuple[ad.AnnData, bool, bool]:
    """Normalize/log using Scanpy and preserve raw counts when available."""
    normalize_total = _resolve_auto_boolean(
        config["normalize_total"], data_kind=data_kind
    )
    do_log1p = _resolve_auto_boolean(config["log1p"], data_kind=data_kind)

    if data_kind == "counts":
        adata.layers["massvision_counts"] = adata.X.copy()

    if normalize_total:
        if adata.X.data.size and np.min(adata.X.data) < 0:
            raise ValueError("Total-count normalization cannot be applied to negative values")
        sc.pp.normalize_total(
            adata,
            target_sum=float(config["target_sum"]),
            exclude_highly_expressed=bool(
                config["exclude_highly_expressed_from_normalization"]
            ),
            max_fraction=float(config["normalization_max_fraction"]),
            inplace=True,
        )

    if do_log1p:
        if adata.X.data.size and np.min(adata.X.data) < 0:
            raise ValueError("log1p cannot be applied to negative expression values")
        sc.pp.log1p(adata)

    return adata, normalize_total, do_log1p


def _calculate_hvgs(
    adata: ad.AnnData,
    config: Mapping[str, Any],
    *,
    data_kind: str,
    n_top: int,
) -> tuple[ad.AnnData, str]:
    """Calculate Scanpy HVGs and return the flavor actually used."""
    flavor = str(config["hvg_flavor"])
    n_top = min(max(int(n_top), 1), adata.n_vars)

    def run(requested_flavor: str) -> None:
        kwargs: dict[str, Any] = {
            "flavor": requested_flavor,
            "n_top_genes": n_top,
            "subset": False,
            "inplace": True,
        }
        if requested_flavor in {"seurat_v3", "seurat_v3_paper"}:
            if data_kind != "counts" or "massvision_counts" not in adata.layers:
                raise ValueError(
                    f"HVG flavor {requested_flavor!r} requires raw counts. "
                    "Select a raw-count matrix/layer or use hvg_flavor='seurat'."
                )
            kwargs["layer"] = "massvision_counts"
        elif requested_flavor not in {"seurat", "cell_ranger"}:
            raise ValueError(
                "Supported hvg_flavor values are 'seurat_v3', "
                "'seurat_v3_paper', 'seurat', and 'cell_ranger'."
            )
        sc.pp.highly_variable_genes(adata, **kwargs)

    try:
        run(flavor)
        resolved_flavor = flavor
    except (ImportError, ModuleNotFoundError) as exc:
        if not bool(config["allow_hvg_fallback"]) or flavor not in {
            "seurat_v3",
            "seurat_v3_paper",
        }:
            raise
        warnings.warn(
            "Scanpy's Seurat v3 HVG calculation could not run, commonly "
            "because scikit-misc is unavailable. Falling back to classic "
            "flavor='seurat' on normalized/log-transformed values.",
            RuntimeWarning,
            stacklevel=2,
        )
        run("seurat")
        resolved_flavor = "seurat"

    if "highly_variable" not in adata.var:
        raise RuntimeError("Scanpy did not create adata.var['highly_variable']")
    return adata, resolved_flavor


def _rank_by_variance(adata: ad.AnnData, n_top: int) -> np.ndarray:
    """Return indices of top-variance genes without densifying the full matrix."""
    matrix = _as_csr(adata.X)
    means = np.asarray(matrix.mean(axis=0)).ravel()
    means_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    variances = np.maximum(means_sq - means**2, 0.0)
    adata.var["massvision_variance"] = variances
    return np.argsort(variances)[::-1][: min(int(n_top), adata.n_vars)]


def _rank_spatial_genes_with_squidpy(
    candidate_adata: ad.AnnData,
    config: Mapping[str, Any],
    *,
    spatial_key: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Rank candidate genes by Moran's I using a Squidpy spatial graph."""
    if sq is None:
        raise ImportError(
            "Spatial gene selection was requested, but Squidpy is not installed. "
            "Install it with 'pip install squidpy' or disable "
            "processing['spatial_gene_selection']."
        )

    coord_type = str(config["spatial_coord_type"])
    common_kwargs: dict[str, Any] = {
        "spatial_key": spatial_key,
        "key_added": "massvision_spatial",
        "copy": False,
    }

    # Squidpy >= 1.7 deprecates the generic spatial_neighbors() dispatcher.
    # Use the explicit graph constructor that matches the acquisition geometry.
    if coord_type == "grid":
        # Standard Visium spots lie on an approximately regular hexagonal grid.
        sq.gr.spatial_neighbors_grid(
            candidate_adata,
            n_neighs=int(config["spatial_n_neighs"]),
            n_rings=int(config["spatial_n_rings"]),
            **common_kwargs,
        )
    elif coord_type == "generic":
        radius = config["spatial_radius"]
        if bool(config["spatial_delaunay"]):
            sq.gr.spatial_neighbors_delaunay(
                candidate_adata,
                radius=radius,
                **common_kwargs,
            )
        elif radius is not None:
            sq.gr.spatial_neighbors_radius(
                candidate_adata,
                radius=radius,
                **common_kwargs,
            )
        else:
            sq.gr.spatial_neighbors_knn(
                candidate_adata,
                n_neighs=int(config["spatial_n_neighs"]),
                **common_kwargs,
            )
    else:
        raise ValueError("spatial_coord_type must be 'grid' or 'generic'")

    stats = sq.gr.spatial_autocorr(
        candidate_adata,
        connectivity_key="massvision_spatial_connectivities",
        genes=list(candidate_adata.var_names),
        mode="moran",
        transformation=True,
        n_perms=config["moran_n_perms"],
        corr_method="fdr_bh",
        attr="X",
        seed=int(config["random_seed"]),
        copy=True,
        n_jobs=int(config["moran_n_jobs"]),
        show_progress_bar=False,
    )

    if stats is None or "I" not in stats.columns:
        raise RuntimeError("Squidpy did not return a valid Moran's I table")

    stats = stats.copy()
    stats.index = stats.index.astype(str)

    keep = np.isfinite(stats["I"].to_numpy(dtype=float))
    minimum_i = config["moran_min_i"]
    if minimum_i is not None:
        keep &= stats["I"].to_numpy(dtype=float) >= float(minimum_i)

    maximum_fdr = config["moran_max_fdr"]
    if maximum_fdr is not None:
        fdr_columns = [
            column
            for column in stats.columns
            if "fdr" in str(column).lower() and str(column).lower().startswith("pval")
        ]
        if not fdr_columns:
            raise ValueError(
                "moran_max_fdr was set, but Squidpy returned no FDR-adjusted "
                "p-value column."
            )
        keep &= stats[fdr_columns[0]].to_numpy(dtype=float) <= float(maximum_fdr)

    filtered = stats.loc[keep].sort_values("I", ascending=False)
    names_to_indices = {name: i for i, name in enumerate(candidate_adata.var_names)}
    indices = np.asarray(
        [names_to_indices[name] for name in filtered.index if name in names_to_indices],
        dtype=int,
    )
    return indices, stats


def _select_genes(
    adata: ad.AnnData,
    config: Mapping[str, Any],
    *,
    data_kind: str,
    spatial_key: str,
) -> tuple[ad.AnnData, str | None, pd.DataFrame | None]:
    """Select genes while always preserving requested named genes."""
    method = str(config["gene_selection"])
    n_top = min(max(int(config["n_top_genes"]), 1), adata.n_vars)

    include_mask, missing = _match_requested_genes(
        adata.var.reset_index(drop=True), config["include_genes"]
    )
    if missing and bool(config["error_on_missing_include_genes"]):
        raise KeyError(f"Requested genes were not found: {missing}")
    if missing:
        warnings.warn(
            f"Requested genes were not found and will be skipped: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )

    resolved_hvg_flavor: str | None = None
    moran_table: pd.DataFrame | None = None

    if method == "all":
        selected = np.arange(adata.n_vars)

    elif method == "gene_list":
        selected = np.flatnonzero(include_mask)
        if selected.size == 0:
            raise ValueError(
                "gene_selection='gene_list' requires at least one valid "
                "processing['include_genes'] entry."
            )

    elif method == "variance":
        selected = _rank_by_variance(adata, n_top)

    elif method in {"hvg", "hvg_then_spatial"}:
        candidate_count = (
            max(n_top, int(config["spatial_candidate_genes"]))
            if method == "hvg_then_spatial" or bool(config["spatial_gene_selection"])
            else n_top
        )
        adata, resolved_hvg_flavor = _calculate_hvgs(
            adata,
            config,
            data_kind=data_kind,
            n_top=candidate_count,
        )
        candidate_indices = np.flatnonzero(
            adata.var["highly_variable"].to_numpy(dtype=bool)
        )

        use_spatial = method == "hvg_then_spatial" or bool(
            config["spatial_gene_selection"]
        )
        if use_spatial:
            candidate = adata[:, candidate_indices].copy()
            spatial_indices, moran_table = _rank_spatial_genes_with_squidpy(
                candidate,
                config,
                spatial_key=spatial_key,
            )
            selected = candidate_indices[spatial_indices[:n_top]]
            if selected.size < n_top:
                warnings.warn(
                    f"Spatial filters retained {selected.size} genes rather than "
                    f"the requested {n_top}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            # Prefer Scanpy's explicit rank when available.
            if "highly_variable_rank" in adata.var:
                ranks = adata.var["highly_variable_rank"].to_numpy(dtype=float)
                valid = np.isfinite(ranks)
                selected = np.flatnonzero(valid)[np.argsort(ranks[valid])][:n_top]
            else:
                selected = candidate_indices[:n_top]

    else:
        raise ValueError(
            "gene_selection must be one of: 'hvg', 'hvg_then_spatial', "
            "'variance', 'all', or 'gene_list'."
        )

    # Always retain explicitly requested genes.
    selected = np.unique(
        np.concatenate([np.asarray(selected, dtype=int), np.flatnonzero(include_mask)])
    )
    if selected.size == 0:
        raise ValueError("No genes were selected")

    selected_adata = adata[:, selected].copy()

    # Add Moran values to selected var metadata when spatial ranking was used.
    if moran_table is not None:
        selected_adata.var["moran_I"] = np.nan
        selected_adata.var["moran_pval"] = np.nan
        selected_adata.var["moran_fdr"] = np.nan
        for gene in selected_adata.var_names:
            if gene not in moran_table.index:
                continue
            selected_adata.var.loc[gene, "moran_I"] = moran_table.loc[gene, "I"]
            pval_cols = [c for c in moran_table.columns if str(c).startswith("pval")]
            if pval_cols:
                selected_adata.var.loc[gene, "moran_pval"] = moran_table.loc[
                    gene, pval_cols[0]
                ]
            fdr_cols = [c for c in pval_cols if "fdr" in str(c).lower()]
            if fdr_cols:
                selected_adata.var.loc[gene, "moran_fdr"] = moran_table.loc[
                    gene, fdr_cols[0]
                ]

    return selected_adata, resolved_hvg_flavor, moran_table


def _scale_selected_genes(
    adata: ad.AnnData,
    config: Mapping[str, Any],
) -> ad.AnnData:
    if not bool(config["scale_genes"]):
        return adata

    sc.pp.scale(
        adata,
        zero_center=bool(config["scale_zero_center"]),
        max_value=(
            None
            if config["scale_max_value"] is None
            else float(config["scale_max_value"])
        ),
        copy=False,
    )
    return adata


# ---------------------------------------------------------------------------
# Spot geometry and rasterization
# ---------------------------------------------------------------------------


def _extract_spot_diameter(
    coordinates_xy: np.ndarray,
    spatial_entry: Mapping[str, Any] | None,
    raster_config: Mapping[str, Any],
) -> tuple[float, str]:
    explicit = raster_config["spot_diameter"]
    if explicit is not None:
        diameter = float(explicit)
        source = "explicit"
    else:
        diameter = math.nan
        source = "missing"

        if spatial_entry is not None:
            scalefactors = spatial_entry.get("scalefactors", {})
            if isinstance(scalefactors, Mapping):
                value = scalefactors.get("spot_diameter_fullres")
                if value is not None:
                    diameter = float(value)
                    source = (
                        "uns['spatial'][library_id]['scalefactors']"
                        "['spot_diameter_fullres']"
                    )

        if not np.isfinite(diameter):
            if not bool(raster_config["allow_diameter_estimation"]):
                raise ValueError(
                    "Spot diameter was not found. Provide "
                    "raster={'spot_diameter': value} in the same units as the "
                    "spatial coordinates."
                )
            if coordinates_xy.shape[0] < 2:
                raise ValueError("At least two spots are needed to estimate diameter")
            tree = cKDTree(coordinates_xy)
            distances, _ = tree.query(coordinates_xy, k=2)
            nearest = distances[:, 1]
            nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
            if nearest.size == 0:
                raise ValueError("Could not estimate nearest-neighbor spot spacing")
            spacing = float(np.median(nearest))
            diameter = spacing * float(
                raster_config["spot_diameter_to_spacing_ratio"]
            )
            source = "estimated_from_median_nearest_neighbor_spacing"
            warnings.warn(
                "spot_diameter_fullres was absent. The spot diameter was "
                f"estimated as {diameter:.3f} coordinate units. Supply an "
                "explicit diameter for exact pathology-image alignment.",
                RuntimeWarning,
                stacklevel=2,
            )

    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Spot diameter must be a positive finite value")
    return diameter, source


def _build_spot_id_map(
    coordinates_xy: np.ndarray,
    spot_diameter: float,
    raster_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rasterize circular footprints while leaving inter-spot gaps invalid."""
    display_diameter = float(spot_diameter) * float(
        raster_config["spot_size_factor"]
    )
    if display_diameter <= 0:
        raise ValueError("spot_size_factor must produce a positive diameter")

    radius = display_diameter / 2.0
    margin = float(raster_config["margin_spot_diameters"]) * display_diameter

    x = coordinates_xy[:, 0]
    y = coordinates_xy[:, 1]

    if bool(raster_config["crop_to_spots"]):
        x0 = float(np.floor(x.min() - radius - margin))
        y0 = float(np.floor(y.min() - radius - margin))
        x1 = float(np.ceil(x.max() + radius + margin))
        y1 = float(np.ceil(y.max() + radius + margin))
    else:
        x0, y0 = 0.0, 0.0
        x1 = float(np.ceil(x.max() + radius + margin))
        y1 = float(np.ceil(y.max() + radius + margin))

    source_width = max(x1 - x0, 1.0)
    source_height = max(y1 - y0, 1.0)

    mode = str(raster_config["resolution_mode"])
    if mode == "min_dimension":
        target = int(raster_config["target_min_dimension"])
        if target <= 0:
            raise ValueError("target_min_dimension must be positive")
        scale = target / min(source_width, source_height)
    elif mode == "coordinate_pixel_size":
        units = raster_config["coordinate_units_per_raster_pixel"]
        if units is None or float(units) <= 0:
            raise ValueError(
                "coordinate_units_per_raster_pixel must be positive when "
                "resolution_mode='coordinate_pixel_size'."
            )
        scale = 1.0 / float(units)
    else:
        raise ValueError(
            "resolution_mode must be 'min_dimension' or "
            "'coordinate_pixel_size'."
        )

    raster_diameter = display_diameter * scale
    minimum_enforced = False
    if bool(raster_config["enforce_min_spot_diameter"]):
        minimum = float(raster_config["min_spot_diameter_px"])
        if raster_diameter < minimum:
            scale = minimum / display_diameter
            raster_diameter = minimum
            minimum_enforced = True

    width = max(1, int(math.ceil(source_width * scale)))
    height = max(1, int(math.ceil(source_height * scale)))
    centers = np.column_stack(((x - x0) * scale, (y - y0) * scale))
    raster_radius = raster_diameter / 2.0

    spot_id_map = np.full((height, width), -1, dtype=np.int32)
    nearest_squared = np.full((height, width), np.inf, dtype=np.float32)

    # With true Visium geometry, footprints do not overlap. If a user increases
    # spot_size_factor for readability and circles overlap, pixels are assigned
    # to the nearest spot center rather than averaged/interpolated.
    for spot_id, (center_x, center_y) in enumerate(centers):
        row0 = max(0, int(math.floor(center_y - raster_radius)))
        row1 = min(height, int(math.ceil(center_y + raster_radius)) + 1)
        col0 = max(0, int(math.floor(center_x - raster_radius)))
        col1 = min(width, int(math.ceil(center_x + raster_radius)) + 1)
        if row0 >= row1 or col0 >= col1:
            continue

        rows, cols = np.ogrid[row0:row1, col0:col1]
        squared = (rows - center_y) ** 2 + (cols - center_x) ** 2
        inside = squared <= raster_radius**2

        local_nearest = nearest_squared[row0:row1, col0:col1]
        update = inside & (squared < local_nearest)
        local_nearest[update] = squared[update]

        local_map = spot_id_map[row0:row1, col0:col1]
        local_map[update] = spot_id

    coordinate_to_raster = np.asarray(
        [
            [scale, 0.0, -x0 * scale],
            [0.0, scale, -y0 * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    raster_to_coordinate = np.asarray(
        [
            [1.0 / scale, 0.0, x0],
            [0.0, 1.0 / scale, y0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    metadata = {
        "coordinate_to_raster": coordinate_to_raster,
        "raster_to_coordinate": raster_to_coordinate,
        "crop_origin_xy": (x0, y0),
        "scale_raster_per_coordinate_unit": scale,
        "physical_spot_diameter_coordinate_units": float(spot_diameter),
        "display_spot_diameter_coordinate_units": display_diameter,
        "display_spot_diameter_raster_px": raster_diameter,
        "minimum_spot_diameter_was_enforced": minimum_enforced,
        "shape_hw": (height, width),
    }
    return spot_id_map, centers, metadata


def _dense_matrix(matrix: Any, dtype: np.dtype) -> np.ndarray:
    if sparse.issparse(matrix):
        dense = matrix.toarray()
    else:
        dense = np.asarray(matrix)
    return np.asarray(dense, dtype=dtype, order="C")


def _materialize_raster_cube(
    spot_matrix: np.ndarray,
    spot_id_map: np.ndarray,
    raster_config: Mapping[str, Any],
) -> tuple[np.ndarray | np.memmap | None, float]:
    height, width = spot_id_map.shape
    n_genes = spot_matrix.shape[1]
    estimated_gb = (
        height
        * width
        * n_genes
        * np.dtype(spot_matrix.dtype).itemsize
        / 1024**3
    )

    if not bool(raster_config["materialize_cube"]):
        return None, float(estimated_gb)

    if estimated_gb > float(raster_config["max_cube_gb"]):
        raise MemoryError(
            f"Requested raster cube is approximately {estimated_gb:.2f} GiB, "
            f"exceeding max_cube_gb={raster_config['max_cube_gb']}. Reduce "
            "target_min_dimension or n_top_genes, increase max_cube_gb, use a "
            "memmap_path, or set materialize_cube=False."
        )

    shape = (height, width, n_genes)
    memmap_path = raster_config["memmap_path"]
    if memmap_path is None:
        cube: np.ndarray | np.memmap = np.empty(shape, dtype=spot_matrix.dtype)
    else:
        path = Path(memmap_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        cube = np.memmap(path, mode="w+", dtype=spot_matrix.dtype, shape=shape)

    cube[...] = raster_config["background_value"]
    flat_ids = spot_id_map.ravel()
    flat_cube = cube.reshape(-1, n_genes)
    valid = flat_ids >= 0
    flat_cube[valid] = spot_matrix[flat_ids[valid]]

    if isinstance(cube, np.memmap):
        cube.flush()
    return cube, float(estimated_gb)


# ---------------------------------------------------------------------------
# Public utilities
# ---------------------------------------------------------------------------


def render_spot_channels(
    spot_matrix: np.ndarray,
    spot_id_map: np.ndarray,
    channel_indices: int | Sequence[int] | np.ndarray,
    *,
    background_value: float = np.nan,
) -> np.ndarray:
    """Render one or more selected channels without allocating the full cube."""
    indices = np.atleast_1d(channel_indices).astype(int)
    if np.any(indices < 0) or np.any(indices >= spot_matrix.shape[1]):
        raise IndexError("channel_indices contains an out-of-range index")

    height, width = spot_id_map.shape
    output = np.full(
        (height, width, len(indices)),
        background_value,
        dtype=spot_matrix.dtype,
    )
    valid = spot_id_map >= 0
    output[valid] = spot_matrix[spot_id_map[valid]][:, indices]
    return output[..., 0] if len(indices) == 1 else output


def spot_ids_from_roi(
    spot_id_map: np.ndarray,
    roi_mask: np.ndarray,
    *,
    selection: str = "intersects",
    minimum_overlap_fraction: float = 0.5,
) -> np.ndarray:
    """Convert a raster ROI into unique measured spot IDs.

    Parameters
    ----------
    spot_id_map:
        Integer footprint map returned by the converter.
    roi_mask:
        Boolean mask with the same height and width.
    selection:
        ``'intersects'`` includes any spot touched by the ROI.
        ``'minimum_overlap'`` includes a spot only when at least the requested
        fraction of its rasterized footprint is inside the ROI.
    minimum_overlap_fraction:
        Used only for ``selection='minimum_overlap'``.
    """
    if roi_mask.shape != spot_id_map.shape:
        raise ValueError("roi_mask and spot_id_map must have identical shapes")

    roi = np.asarray(roi_mask, dtype=bool)
    touched = spot_id_map[roi & (spot_id_map >= 0)]
    if touched.size == 0:
        return np.empty(0, dtype=int)

    if selection == "intersects":
        return np.unique(touched)
    if selection != "minimum_overlap":
        raise ValueError("selection must be 'intersects' or 'minimum_overlap'")

    threshold = float(minimum_overlap_fraction)
    if not 0 <= threshold <= 1:
        raise ValueError("minimum_overlap_fraction must be between 0 and 1")

    selected: list[int] = []
    for spot_id in np.unique(touched):
        footprint = spot_id_map == spot_id
        overlap = np.count_nonzero(footprint & roi)
        total = np.count_nonzero(footprint)
        if total > 0 and overlap / total >= threshold:
            selected.append(int(spot_id))
    return np.asarray(selected, dtype=int)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------


def h5ad_to_spot_footprint_cube(
    h5ad_path: str | Path,
    *,
    processing: Mapping[str, Any] | None = None,
    raster: Mapping[str, Any] | None = None,
) -> SpotFootprintRasterResult:
    """Convert a spatial ``.h5ad`` into processed spots and a circular raster.

    Parameters
    ----------
    h5ad_path:
        Input AnnData file.
    processing:
        Flat dictionary overriding ``DEFAULT_PROCESSING``. Unknown keys raise an
        error, which helps catch UI or spelling mistakes.
    raster:
        Flat dictionary overriding ``DEFAULT_RASTER``.

    Returns
    -------
    SpotFootprintRasterResult
        Processed AnnData, spot matrix, gene metadata, circular raster cube,
        spot ID map, valid mask, coordinate transforms, and diagnostics.
    """
    pconf = _merge_config(
        DEFAULT_PROCESSING, processing, config_name="processing"
    )
    rconf = _merge_config(DEFAULT_RASTER, raster, config_name="raster")

    path = Path(h5ad_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    source_adata = ad.read_h5ad(path)

    # Resolve the optional Visium library metadata before subsetting.
    library_id, spatial_entry = _resolve_library_metadata(
        source_adata, pconf["library_id"]
    )

    # Select one library from a concatenated AnnData object when requested.
    obs_library_key = pconf["obs_library_key"]
    if library_id is not None and obs_library_key is not None:
        if obs_library_key not in source_adata.obs:
            raise KeyError(
                f"obs_library_key={obs_library_key!r} was not found in adata.obs"
            )
        library_mask = (
            source_adata.obs[obs_library_key].astype(str).to_numpy()
            == str(library_id)
        )
        if not np.any(library_mask):
            raise ValueError(
                f"No spots matched {obs_library_key} == {library_id!r}"
            )
        source_adata = source_adata[library_mask].copy()

    spatial_key = str(pconf["spatial_key"])
    if spatial_key not in source_adata.obsm:
        raise KeyError(
            f"adata.obsm[{spatial_key!r}] was not found. "
            f"Available keys: {list(source_adata.obsm.keys())}"
        )

    coordinates = np.asarray(source_adata.obsm[spatial_key], dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must have shape (n_spots, >=2); "
            f"got {coordinates.shape}"
        )
    coordinates = coordinates[:, :2]
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Spatial coordinates contain NaN or infinite values")

    matrix, source_var, matrix_source = _resolve_expression_source(
        source_adata, str(pconf["matrix_source"])
    )

    # gene_table = _build_gene_table(source_var)

    # # Construct a clean working AnnData whose var_names are unique display
    # # names. Keep original IDs/symbols in var columns.
    # working = ad.AnnData(
    #     X=matrix,
    #     obs=source_adata.obs.copy(),
    #     var=gene_table.copy(),
    #     obsm={spatial_key: coordinates.copy()},
    # )
    # working.var_names = gene_table["gene_name"].astype(str).to_numpy()
    # working.obs_names = source_adata.obs_names.astype(str)

    gene_table = _build_gene_table(source_var)

    # Prepare AnnData-compatible string indices before construction.
    gene_table.index = pd.Index(
        gene_table["gene_name"].astype(str),
        name="gene_name",
    )

    obs_table = source_adata.obs.copy()
    obs_table.index = pd.Index(
        source_adata.obs_names.astype(str),
        name=source_adata.obs.index.name,
    )

    working = ad.AnnData(
        X=matrix,
        obs=obs_table,
        var=gene_table,
        obsm={spatial_key: coordinates.copy()},
    )

    # Preserve spatial metadata when available; Squidpy and plotting tools may
    # use it. A deep copy is not necessary for read-only nested metadata here.
    if "spatial" in source_adata.uns:
        working.uns["spatial"] = source_adata.uns["spatial"]

    # Tissue filtering before QC.
    if bool(pconf["tissue_only"]) and "in_tissue" in working.obs:
        tissue_mask = _truthy_tissue_mask(working.obs["in_tissue"])
        if not np.any(tissue_mask):
            raise ValueError("No observations were marked as in_tissue")
        working = working[tissue_mask].copy()

    # Infer/validate data type.
    data_kind = str(pconf["data_kind"])
    if data_kind == "auto":
        data_kind = "counts" if _looks_like_counts(_as_csr(working.X)) else "processed"
    if data_kind not in {"counts", "processed"}:
        raise ValueError("data_kind must be 'auto', 'counts', or 'processed'")

    if data_kind == "processed":
        warnings.warn(
            "The selected matrix does not appear to contain raw counts. "
            "Automatic normalization/log1p are disabled, and Seurat v3 HVG "
            "selection may not be appropriate. Confirm matrix_source and "
            "processing settings for this dataset.",
            RuntimeWarning,
            stacklevel=2,
        )

    _annotate_gene_groups(working, pconf)
    working = _filter_spots_with_qc(working, pconf)
    working = _filter_genes_with_scanpy(working, pconf)

    working, normalized, logged = _normalize_with_scanpy(
        working,
        pconf,
        data_kind=data_kind,
    )

    # For processed matrices, an explicitly requested Seurat v3 HVG would fail.
    # In automatic/default use, fall back to variance, which is a safer generic
    # feature-ranking method for unknown transformed values.
    resolved_gene_selection = str(pconf["gene_selection"])
    resolved_config = dict(pconf)
    if (
        data_kind == "processed"
        and resolved_gene_selection in {"hvg", "hvg_then_spatial"}
        and str(pconf["hvg_flavor"]) in {"seurat_v3", "seurat_v3_paper"}
    ):
        warnings.warn(
            "Processed data were selected with a raw-count HVG flavor. "
            "Switching gene_selection to 'variance'. Set hvg_flavor='seurat' "
            "explicitly to use Scanpy HVG selection on processed/log values.",
            RuntimeWarning,
            stacklevel=2,
        )
        resolved_config["gene_selection"] = "variance"
        resolved_gene_selection = "variance"

    selected, resolved_hvg_flavor, moran_table = _select_genes(
        working,
        resolved_config,
        data_kind=data_kind,
        spatial_key=spatial_key,
    )
    selected = _scale_selected_genes(selected, pconf)

    dtype = np.dtype(str(pconf["dtype"]))
    spot_matrix = _dense_matrix(selected.X, dtype)

    selected_coordinates = np.asarray(selected.obsm[spatial_key], dtype=float)[:, :2]
    spot_diameter, diameter_source = _extract_spot_diameter(
        selected_coordinates,
        spatial_entry,
        rconf,
    )
    spot_id_map, raster_centers, transform = _build_spot_id_map(
        selected_coordinates,
        spot_diameter,
        rconf,
    )
    raster_cube, estimated_cube_gb = _materialize_raster_cube(
        spot_matrix,
        spot_id_map,
        rconf,
    )

    # Build user-friendly output tables.
    gene_output = selected.var.copy()
    gene_output.insert(0, "selected_gene_name", selected.var_names.astype(str))
    gene_output["selected_channel_index"] = np.arange(selected.n_vars, dtype=int)

    spot_output = selected.obs.reset_index().rename(
        columns={selected.obs.index.name or "index": "barcode"}
    )
    if "barcode" not in spot_output:
        spot_output.insert(0, "barcode", selected.obs_names.astype(str))
    spot_output["source_x"] = selected_coordinates[:, 0]
    spot_output["source_y"] = selected_coordinates[:, 1]
    spot_output["raster_x"] = raster_centers[:, 0]
    spot_output["raster_y"] = raster_centers[:, 1]
    spot_output["spot_matrix_row"] = np.arange(selected.n_obs, dtype=int)

    metadata: dict[str, Any] = {
        "source_h5ad": str(path),
        "matrix_source": matrix_source,
        "data_kind": data_kind,
        "normalization_applied": normalized,
        "log1p_applied": logged,
        "gene_selection_requested": pconf["gene_selection"],
        "gene_selection_resolved": resolved_gene_selection,
        "hvg_flavor_requested": pconf["hvg_flavor"],
        "hvg_flavor_resolved": resolved_hvg_flavor,
        "squidpy_spatial_selection_used": moran_table is not None,
        "spatial_key": spatial_key,
        "library_id": library_id,
        "spot_diameter_source": diameter_source,
        "spot_diameter_coordinate_units": float(spot_diameter),
        "processing": dict(pconf),
        "processing_resolved": dict(resolved_config),
        "raster": dict(rconf),
        "transform": transform,
        "n_spots": int(selected.n_obs),
        "n_genes": int(selected.n_vars),
        "raster_shape_hw": tuple(spot_id_map.shape),
        "estimated_cube_gb": estimated_cube_gb,
        "background_is_unmeasured": True,
        "no_spatial_interpolation": True,
        "coordinate_warning": (
            "AnnData does not enforce spatial coordinate units. Confirm that "
            "spot diameter and coordinates use the same coordinate system."
        ),
    }

    return SpotFootprintRasterResult(
        processed_adata=selected,
        spot_matrix=spot_matrix,
        gene_names=selected.var_names.to_numpy(dtype=object),
        gene_ids=selected.var["gene_id"].astype(str).to_numpy(dtype=object),
        gene_table=gene_output,
        barcodes=selected.obs_names.to_numpy(dtype=object),
        spot_table=spot_output,
        raster_cube=raster_cube,
        spot_id_map=spot_id_map,
        valid_mask=spot_id_map >= 0,
        raster_spot_centers_xy=raster_centers,
        metadata=metadata,
    )
