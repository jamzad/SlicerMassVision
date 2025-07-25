Getting Started
===============

Installing 3D Slicer
--------------------
MassVision is hosted in 3D Slicer image computation platform. Please ensure that you installed the latest stable version of 3D Slicer on your system. The detailed instruction for installation and system requirements can be found on the official `3D Slicer Installation Guide <https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#installing-3d-slicer>`_.To learn more about the platform and its applications, please refer to the official `3D Slicer Documentation <https://slicer.readthedocs.io/en/latest/>`_.


.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/SlicerLogo.png
      :width: 150
      :align: center

Installing MassVision
---------------------
#. Open 3D Slicer. 
#. Select "Extensions Manager" from the "View" menu at the top of the screen.

    .. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ExtensionsManager.PNG
        :width: 250

#. Search for "MassVision" in the top right box and install the extension.

    .. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/MassVisionInstall.png
        :width: 400


#. The 3D Slicer should be restarted to complete the installation.

Opening MassVision
------------------
#. Open 3D Slicer
#. Go to the 'Modules' section of the toolbar (top left) and click on the magnifying glass |ModulesIcon|.
#. Search for "MassVision" and switch to the module.  


    .. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ModuleFinder.png
        :width: 600

    .. |ModulesIcon| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ModulesIcon.png
                        :height: 30


#. Once the module is open, you have access to all tabs for performing your analysis: 

    .. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/MassVisionHome.png
        :width: 400
        :align: center

Test Data
---------
 
MassVision provides users with test data for exploring and evaluating its functionalities. The data can be downloaded from `here <https://github.com/jamzad/SlicerMassVision/releases/tag/test-data>`_

The test data includes a sample **MSI data** in both structured CSV and hierarchical HDF5 format, along with the corresponding **histopathology image** in TIFF format. The data is collected using DESI modality from a colorectal tissue slide as part of the following study:

Kaufmann M, Iaboni N, Jamzad A, Hurlbut D, Ren KYM, Rudan JF, Mousavi P, Fichtinger G, Varma S, Caycedo-Marulanda A, et al. Metabolically Active Zones Involving Fatty Acid Elongation Delineated by DESI-MSI Correlate with Pathological and Prognostic Features of Colorectal Cancer. Metabolites. 2023; 13(4):508. https://doi.org/10.3390/metabo13040508