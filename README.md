# MassVis*ion*

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/MassVision.png" alt="logo" width="40%"/>
</p>

MassVis*ion* is a software solution developed in 3D Slicer platform for end-to-end AI-driven analysis of Mass Spectrometry Imaging (MSI) data.

The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), dataset curation with spatial- and spectral-guidance, multi-slide dataset merge via feature alignment, denoising via spatial aggregation, AI model training and validation, and whole-slide AI deployment. 


<img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot1.png" alt="screenshot" width="100%"/>

## Features
* Data format: imzML, structured CSV, hierarchical HDF5, DESI TXT images 
* Visualization: targeted (single-ion heatmap, multi-ion colormap), untargeted (global contrast PCA, local contrast PCA, UMAP, t-SNE), segmentation, pixel spectrum
* Dataset generation: spatial co-localization to pathological annotation, spatial annotation and labelled ROI extraction
* Multi-slide merge: feature alignment, peaks matching
* Preprocessing: normalization (TIC, TSC, ingle-ion, mean, median, RMS), subband selection, spatial pixel aggregation
* Statistical analysis: ANOVA, Volcano, Boxplot
* AI Model training: feature ranking, feature selection, data stratification, data balancing, model selection, cross validation
* Whole-slide AI deployment: global deployment, masked deployment

## Demonstration
* Visualization with global vs local contrast [demo](https://www.dropbox.com/scl/fi/tiwy6mm8pompeeprexf0l/visualization.gif?rlkey=tqtly8rqeymvxkhmdf9hf4039&st=anz46hn1&raw=1)
* Spatial colocalization of mass spectrometry visualization with histopathology annotations [demo](https://www.dropbox.com/scl/fi/cumbv2xfwfgixyxdhuqxz/registration.gif?rlkey=cvi87xl1jz5l9y1vn2te4ktru&st=6fxm9mkb&raw=1)
* ROI selection with pathology guide for dataset generation [demo](https://www.dropbox.com/scl/fi/03ff1aci9qgbgr735k9up/roiselection.gif?rlkey=7sb5fvcdh12g2ra7jnr3x2n2f&st=wqfd5fht&raw=1)

## Installation
Visit [here](https://SlicerMassVision.readthedocs.io/en/latest/Getting%20Started.html) for a complete installation guide from Extentision Manager within 3D Slicer

## Documentations
Visit [here](https://SlicerMassVision.readthedocs.io/en/latest/) to access user guide, tutorials, and other documentations.

## Test Data
MassVis*ion* provides users with a test data for exploring and evaluating its functionalities. The test data includes a sample **MSI data** along with the corresponding **histopathology image** in TIFF format. The data is available in the [release section](https://github.com/jamzad/SlicerMassVision/releases/tag/test-data)

## Citation
Please use the following citations if you use MassVis*ion* in your studies and publication

Jamzad, A.; Warren, J.; Syeda, A.; Kaufmann, M.; Iaboni, N.; Nicol, C.; Rudan, J.; Ren, K.; Hurlbut, D.; Varma, S.; Fichtinger, G.; Mousavi, P. MassVision: An Open-Source End-to-End Platform for AI-Driven Mass Spectrometry Image Analysis. bioRxiv 2025. [https://doi.org/10.1101/2025.01.29.635489](https://doi.org/10.1101/2025.01.29.635489). 
