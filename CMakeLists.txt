cmake_minimum_required(VERSION 3.13.4)

project(MassVision)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://slicermassvision.readthedocs.io/")
set(EXTENSION_CATEGORY "Spectral Imaging")
set(EXTENSION_CONTRIBUTORS "Amoon Jamzad (Med-i Lab, Queen's University), Jade Warren, Ayesha Syeda")
set(EXTENSION_DESCRIPTION "MassVision is a 3D Slicer module developed for end-to-end analysis of Mass Spectrometry Imaging (MSI) data. The functionalities include data exploration via various targeted and non-targeted visualization, co-localization to spatial labels (histopathology annotations), spatial annotation for dataset generation with spatial- and spectral-guidance, multi-slide data aggregation via feature alignment, spectral and spatial denoising and preprocessing, statistical analysis, supervised AI model training, and whole-slide model deployment.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/MassVision.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot1.png"; "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot2.png"; "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot3.png"; "https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/Screenshot4.png")
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
