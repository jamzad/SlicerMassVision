# MassVis*ion*

<p align="center">
  <img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/MassVision.png" alt="logo" width="40%"/>
</p>

MassVis*ion* is a software solution developed in 3D Slicer platform for end-to-end analysis of Mass Spectrometry Imaging (MSI) data, particularly Desorption ElectroSpray Ionization (DESI) modality. 

The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), dataset generation with spatial- and spectral-guidance, multi-slide data aggregation via feature alignment, denoising via spatial aggregation, machine learning model training, and whole-slide model deployment. 


<img src="https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot.png" alt="screenshot" width="100%"/>

## Features
* Data format: DESI TXT images, structured CSV MSI
* Visualization: targeted (single-ion heatmap, multi-ion colormap), untargeted (global PCA, regional PCA), pixel spectrum
* Dataset generation: spatial colocalization to pathological annotation, labelled ROI extraction
* Multi-slide peak alignment and dataset merge
* Preprocessing: normalization (TIC, single-ion), subband selection, spatial pixel aggregation
* Model training: data stratification, data balancing, model selection
* Whole-slide deployment: global deployment, masked deployment

## Demonstration
* Visualization with global vs local contrast [demo](https://www.dropbox.com/scl/fi/tiwy6mm8pompeeprexf0l/visualization.gif?rlkey=tqtly8rqeymvxkhmdf9hf4039&st=anz46hn1&raw=1)
* Spatial colocalization of mass spectrometry visualization with histopathology annotations [demo](https://www.dropbox.com/scl/fi/cumbv2xfwfgixyxdhuqxz/registration.gif?rlkey=cvi87xl1jz5l9y1vn2te4ktru&st=6fxm9mkb&raw=1)
* ROI selection with pathology guide for dataset generation [demo](https://www.dropbox.com/scl/fi/03ff1aci9qgbgr735k9up/roiselection.gif?rlkey=7sb5fvcdh12g2ra7jnr3x2n2f&st=wqfd5fht&raw=1)

## Installation
Visit [here](https://slicerdesi.readthedocs.io/en/latest/Installation.html) for a complete installation guide from Extentision Manager within 3D Slicer

## Documentations
Visit [here](https://slicerdesi.readthedocs.io/) to access user guide, tutorials, and other documentations.


## Citation
Please use the following citations if you are using MassVis*ion*
* TBA 
