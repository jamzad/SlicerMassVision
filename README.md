# SlicerDESI
<p align="center">
  <img src="https://github.com/jamzad/SlicerDESI/blob/main/SlicerDESI.png" alt="logo" width="35%"/>
</p>


SlicerDESI is a module developed in 3D Slicer platform for end-to-end analysis of Mass Spectrometry Imaging (MSI) data, particularly Desorption ElectroSpray Ionization (DESI) modality. The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), dataset generation with spatial- and spectral-guidance, multi-slide data aggregation via feature alignment, denoising via spatial aggregation, machine learning model training, and whole-slide model deployment. 


<img src="https://github.com/jamzad/SlicerDESI/blob/main/Screenshot.png" alt="screenshot" width="100%"/>

## Features
* Data format: DESI TXT images, structured CSV MSI
* Visualization: targeted (single-ion heatmap, multi-ion colormap), untargeted (global PCA, regional PCA, pixel spectrum)
* Dataset generation: spatial colocalization to pathological annotation, labelled ROI extraction
* Multi-slide peak alignment and dataset merge
* Preprocessing: normalization (TIC, single-ion), subband selection, spatial pixel aggregation
* Model training: data stratification, data balancing, model selection
* Whole-slide deployment: global deployment, masked deployment

## Citation
Please use the following citations if you are using SlicerDESI
* TBA
