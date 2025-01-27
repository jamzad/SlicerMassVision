Importing Data
=====

To load a new MSI data or a MassVision project, navigate to the **Data Import** tab.

Clear Scene
----------
To clear any open data and analysis in MassVision, click on 'Clear data and analysis' at the top of the tab.

.. tip::
   We recommend that users clear MassVision before uploading any new MSI data, or starting a new independent task. 

Load Existing Project
-----------
If you have a saved MassVision project you would like to resume working on, click 'Load project.' at the top of the tab. 

.. important::
   After loading a saved MassVision project, you must reload the corresponding MSI data to resume editing from where you left off.

MassVision projects contain all processed files, such as annotated scenes, labeled segments, and visualizations. However, the original MSI data used in the analysis is excluded to prevent project size inflation and avoid duplicating the original data.


Import MSI Data
-------
#. Click 'Select file...' under MSI data. A file dialog box will open, allowing you to browse and select the file. The current version of MassVision supports modality independent structured MSI files (.csv), hierarchical HDF5 files (.h5), and DESI MSI image files (.txt). Select the format from the dropdown menu, select the file, and click 'Open' to load the data.

.. note::
   The instruction on how to structure your MSI data into CSV or HDF5 compatible with MassVision can be found in `Data Structure <https://slicermassvision.readthedocs.io/en/latest/Data%20Structure.html#data-structure>`_ page.

#. After the data is loaded, general information like number of pixels, number of ions, and the name and location of the file will appears in the text box. The TIC visualization will also be displayed in the viewer. To adjust the image's contrast, click the |WinLevIcon| icon in the Slicer toolbar at the top of the screen. You can adjust the contrast by left-click-and-drag on the image or by selecting a region.  

.. |WinLevIcon| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/AdjustWindowLevel.png
   :height: 30

.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ImportMSIFile.png
    :width: 600



Import Reference Image
---------
To load a gold-standard image like histopathology annotations, click 'Select file...' under Reference Image. A file dialog box will appear, allowing you to upload the desired image. 

.. image:: https://raw.githubusercontent.com/jamzad/SlicerDESI/main/docs/source/Images/LoadPathology.png
    :width: 600



