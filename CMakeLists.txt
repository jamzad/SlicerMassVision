cmake_minimum_required(VERSION 3.13.4)

project(SlicerDESI)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/jamzad/SlicerDESI")
set(EXTENSION_CATEGORY "Mass Spectrometry")
set(EXTENSION_CONTRIBUTORS "Amoon Jamzad, Jade Warren, Ayesha Syeda (Med-i Lab, Queen's University)")
set(EXTENSION_DESCRIPTION "SlicerDESI is a module developed in 3D Slicer platform for end-to-end analysis of Mass Spectrometry Imaging (MSI) data, particularly Desorption ElectroSpray Ionization (DESI). The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), dataset generation with spatial- and spectral-guidance, multi-slide data aggregation via feature alignment, denoising via spatial aggregation, machine learning model training, and whole-slide model deployment.")
set(EXTENSION_ICONURL "https://github.com/jamzad/SlicerDESI/blob/main/SlicerDESI.png")
set(EXTENSION_SCREENSHOTURLS "NA")
set(EXTENSION_DEPENDS "NA") 

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SlicerDESI)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
