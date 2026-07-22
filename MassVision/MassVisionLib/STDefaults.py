
# ---------------------------------------------------------------------------
# User-facing configuration - default parameters for spatial transcriptomics 
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
    "min_genes_per_spot": 0,  ##
    "max_counts_per_spot": None,
    "max_genes_per_spot": None,
    "max_pct_mito": None,

    # Gene quality control ------------------------------------------------
    "min_spots_per_gene": 10,  ##
    "min_total_counts_per_gene": 0,

    # Mitochondrial/ribosomal annotations are useful QC fields. They are not
    # removed by default because exclusion is study-dependent.
    "mitochondrial_prefixes": ["MT-"],
    "ribosomal_prefixes": ["RPL", "RPS"],
    "exclude_mitochondrial_genes": True,
    "exclude_ribosomal_genes": True,
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
    "n_top_genes": 1_000,  ##

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
    "target_min_dimension": 400,
    "coordinate_units_per_raster_pixel": None,

    # Spot geometry -------------------------------------------------------
    # If None, attempt to read spot_diameter_fullres. Final fallback estimates
    # diameter from nearest-neighbor spacing using the 55/100 Visium ratio.
    "spot_diameter": None,
    "allow_diameter_estimation": True,
    "spot_diameter_to_spacing_ratio": 0.55,

    # 1.0 corresponds to the inferred/physical capture footprint. Larger or
    # smaller values are conventional display scaling, not interpolation.
    "spot_size_factor": 1.3,

    # Preserve true scaled size by default. Enable only for readability at very
    # low raster resolutions; doing so changes the displayed footprint size.
    "enforce_min_spot_diameter": False,
    "min_spot_diameter_px": 3.0,

    # Canvas --------------------------------------------------------------
    "crop_to_spots": True,
    "margin_spot_diameters": 0.0,
    "background_value": None,

    # Dense cube allocation ----------------------------------------------
    "materialize_cube": True,
    "max_cube_gb": 2.0,
    "memmap_path": None,
}

