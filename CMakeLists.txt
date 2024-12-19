cmake_minimum_required(VERSION 3.13.4)

project(MassVision)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://slicermassvision.readthedocs.io/")
set(EXTENSION_CATEGORY "Spectral Imaging")
set(EXTENSION_CONTRIBUTORS "Med-i Lab, Queen's University (Amoon Jamzad, Jade Warren, Ayesha Syeda)")
set(EXTENSION_DESCRIPTION "MassVision is a module developed in 3D Slicer platform for end-to-end analysis of Mass Spectrometry Imaging (MSI) data. The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), dataset generation with spatial- and spectral-guidance, multi-slide data aggregation via feature alignment, denoising via spatial aggregation, machine learning model training, and whole-slide model deployment.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/MassVision.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot.png")
set(EXTENSION_DEPENDS "NA") 

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MassVision)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
