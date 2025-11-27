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

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/refs/heads/main/docs/source/Images/functions.png" alt="func" width="100%"/>
</p>

## Demonstration
Visualization and exploration

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/refs/heads/main/docs/source/Images/visualization.gif" width="100%"/>
</p>

Spatial colocalization

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/refs/heads/main/docs/source/Images/colocalization.gif" width="100%"/>
</p>

Pathology guided ROI dataset curation

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/refs/heads/main/docs/source/Images/roi.gif" width="100%"/>
</p>

Statustucal analysis

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/refs/heads/main/docs/source/Images/statistical.gif" width="100%"/>
</p>

## Installation
Visit [here](https://SlicerMassVision.readthedocs.io/en/latest/Getting%20Started.html) for a complete installation guide within 3D Slicer

## Documentations
Visit [here](https://SlicerMassVision.readthedocs.io/en/latest/) to access user guide, tutorials, and other documentations.

## Test Data
MassVis*ion* provides users with a test data for exploring and evaluating its functionalities. The test data includes a sample **MSI data** in structured CSV and hierarchical HDF5 format, along with the corresponding **histopathology image** in TIFF format. The data is available in the [release section](https://github.com/jamzad/SlicerMassVision/releases/tag/test-data). The data is collected using DESI modality from a colorectal tissue slide. The raw dataset is released with MassVis*ion* and available in MetaboLights with the identifier [MTBLS12868](https://www.ebi.ac.uk/metabolights/editor/MTBLS12868/) .

## Citation
Please use the following citations if you use MassVis*ion* in your studies and publication

Jamzad, A.; Warren, J.; Syeda, A.; Kaufmann, M.; Iaboni, N.; Nicol, C.; Rudan, J.; Ren, K.; Hurlbut, D.; Varma, S.; Fichtinger, G.; Mousavi, P. MassVision: An Open-Source End-to-End Platform for AI-Driven Mass Spectrometry Imaging Analysis. Analytical Chemistry 2025. [https://doi.org/10.1021/acs.analchem.5c04018](https://doi.org/10.1021/acs.analchem.5c04018). 
