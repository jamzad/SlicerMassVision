EmbedVision
###########

.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/embedvision_logo.png
    :width: 300
    :align: center
    :alt: EmbedVision logo.

**EmbedVision** is a mode within MassVision for interactive exploration of channel-rich spatial data. It extends the spatial visualization and analysis tools originally developed for mass spectrometry imaging to learned feature maps and other high-dimensional spatial tensors.

Examples include vision-foundation-model embeddings, intermediate neural-network feature maps, spatial transcriptomics, hyperspectral data, and other measurements in which every spatial location is represented by a vector of values. EmbedVision treats these inputs using a common tensor representation and provides tools for global visualization, region-aware probing, similarity analysis, clustering, feature inspection, and comparison with a reference image.


Spatial Tensor Representation
*****************************

EmbedVision represents an input as a three-dimensional tensor with shape ``H x W x C``:

* ``H`` and ``W`` define the spatial grid.
* ``C`` defines the number of channels or features at each spatial location.

For learned representations, the channels are embedding dimensions or latent features. For measured data, the channels may represent genes, molecular measurements, spectral bands, or other recorded variables. The channel meanings may differ, but the spatial data structure is the same.


.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/EmbedVisionDataArray.png
    :width: 600
    :align: center


Starting EmbedVision
********************

Select **EmbedVision** using the Switch Mode section on the **Home** tab. You can return to the **MassVision** mode using the same process.

A typical workflow is:

1. Import a channel-rich spatial tensor.
2. Optionally load its corresponding reference image.
3. Generate a global representation using **PCA (global contrast)**.
4. Select a spatial prompt and generate **P3CA (local contrast)**.
5. Explore the representation using similarity maps, clustering, nonlinear visualization, and channel-contribution summaries.
6. Compare outputs using overlay or wipe visualization and save useful projection transforms for later use.

Input Data
**********

Standard spatial tensors
------------------------

EmbedVision supports spatial tensors stored as NumPy arrays, hierarchical HDF5 files, and structured CSV files. Select the data file in the **Data Import** tab and load it using the same general workflow used for MassVision data.

The recommended representation is a numeric array with shape ``H x W x C``. Missing or unmeasured spatial locations should be represented consistently using the background or validity-mask convention appropriate for the dataset.

See `Data Structure <https://slicermassvision.readthedocs.io/en/latest/Data%20Structure.html>`_ for the general organization of spatial and channel dimensions used by MassVision.

Reference images
----------------

An optional reference image can be loaded with the tensor. Examples include the original natural image, an H&E slide, an ultrasound image, or another co-registered modality. The reference image can be viewed alongside the representation maps and used with the blending tools for visual comparison.

The spatial tensor and reference image should describe the same field of view. When their resolutions differ, confirm that the spatial correspondence is correct before interpreting local prompts or similarity maps.

Spatial transcriptomics
-----------------------

EmbedVision also supports the import of AnnData or Visium-style ``.h5ad`` files. The spatial-transcriptomics importer can preprocess the spot-by-gene matrix, select informative genes, and create a regular image-compatible tensor by rasterizing the measured spot footprints.

Each raster pixel within a measured spot receives the same expression vector as that spot. Values are not interpolated between spots, and unmeasured areas remain background. This preserves the original spot-level measurements while allowing the data to be explored using the same visualization tools as other spatial tensors.

.. important::
    Raster pixels within the same transcriptomic spot are repeated visual representations of one measurement. They should not be treated as independent biological observations in downstream statistical analysis.

Global Representation
*********************

PCA (global contrast)
---------------------

Click **PCA (global contrast)** to summarize the full spatial tensor using principal component analysis. The projection basis is estimated from all valid spatial locations, and the first three principal components are displayed as the red, green, and blue channels of an RGB representation map.

Global PCA provides a useful overview of the dominant variation in the complete tensor. However, large background regions, common tissue types, acquisition effects, or broad anatomical structures may dominate the projection and suppress variation within a smaller region of interest.

Position-Prompted PCA
*********************

P3CA (local contrast)
---------------------

**Position-prompted PCA (P3CA)** provides a region-aware view of the spatial representation. Instead of estimating the projection from the full tensor, P3CA estimates its normalization and principal directions using the feature vectors inside a user-selected spatial prompt.

To generate a P3CA representation:

1. Click the region-selection control for local contrast.
2. Draw or define a spatial region over the reference image or an existing representation map.
3. Adjust the region if needed.
4. Click **P3CA (local contrast)**.

The resulting projection is applied to the entire spatial tensor, not only to the selected region. The output therefore acts as a local representation lens: it shows where the feature directions that vary most strongly within the selected region are expressed throughout the full image.


P3CA is label-free and encoder-agnostic. It does not retrain or modify the model that generated the embeddings and can also be applied to measured spatial tensors such as gene-expression data.

.. tip::
    Compare the P3CA map with the global PCA map. A useful prompt may reveal boundaries, subregions, or repeated structures that are weak or absent in the global representation.

.. note::
    P3CA deliberately emphasizes variation associated with the selected region. Unrelated areas may become less contrasted or saturated because the projection is locally optimized rather than globally balanced.

Similarity Maps
***************

Similarity analysis shows where feature vectors resemble a selected spatial query.

* Place a query point on the displayed image and generate a **Similarity Heatmap** to compare that location with the rest of the tensor.
* Use a selected region or segmentation as a query when region-based similarity is more appropriate.
* Use the similarity thumbnail view to inspect the query together with the resulting similarity pattern.

Similarity maps can help determine whether a local representation pattern is unique, repeated elsewhere, or shared across an anatomical or pathological structure. The result depends on the current tensor, preprocessing, and selected similarity measure.

Nonlinear Visualization
************************

EmbedVision supports **UMAP** and **t-SNE** for nonlinear dimensionality reduction. These methods can reveal representation structure that is not well summarized by a linear projection.

Select the desired method, configure its parameters, and generate the visualization. UMAP and t-SNE are generally slower than PCA and may change when their parameters or random initialization are changed. They are most useful as complementary exploratory views rather than fixed coordinate systems for direct cross-sample comparison.

Clustering and Feature Contributions
************************************

Clustering
----------

Use clustering to group spatial locations with similar high-dimensional signatures. The output is displayed as a label map in which locations assigned to the same cluster share a color.

Clustering provides a segmentation-like view of representation patterns, but the clusters are data-driven and should not automatically be interpreted as biological or anatomical classes.

Dominant channels and features
------------------------------

Thumbnail and contribution views can be used to inspect the channels that contribute most strongly to a projection or that are most associated with a selected cluster.

For measured tensors, the channels may have explicit names such as genes, ions, or spectral bands. For learned embeddings, an individual channel is usually a latent dimension and may not have a direct standalone interpretation. In that setting, channel rankings are most useful for comparing prompts, encoders, or samples rather than assigning semantic meaning to one dimension in isolation.

Reference-Image Comparison
**************************

Use the blending controls to compare a representation map with its reference image or with another generated map. Available views include:

* **Overlay:** Blends foreground and background images with an adjustable opacity.
* **Wipe:** Reveals the two images on opposite sides of a movable boundary.
* **Vertical wipe:** Uses a vertical comparison boundary.

These tools are useful for checking spatial correspondence and relating representation patterns to visible morphology without permanently combining the images.

Saving and Reusing Projections
******************************

EmbedVision can save a fitted PCA or P3CA projection and load it again later. A saved projection preserves the fitted transformation so that the same representation lens can be reapplied without estimating a new basis.

This is useful for:

* applying one prompted projection to additional samples;
* comparing multiple images from the same encoder;
* reproducing a previously generated representation;
* separating projection fitting from later visualization.

.. important::
    Reuse a saved projection only with tensors from the same feature space and compatible preprocessing. Embeddings from different model layers, encoders, feature dimensions, or normalization pipelines are not directly interchangeable.



