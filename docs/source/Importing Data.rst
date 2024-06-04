Importing Data
=====

To load the data navigate to the **Data Import** tab. You have different options for data format. 

Import MSI Data
-------
#. To import MSI data, click 'Select file...' under MSI data. A file dialog box will open, allowing you to browse and select the file. The current supported formats are DESI image files (.txt) and tabular structured files (.csv). After choosing the file, the path of the selected file will be displayed.
#. After the data is loaded, the geneerated TIC image will be displayed in the viewer. To adjust the image's contrast, click the |WinLevIcon| icon in the Slicer toolbar at the top of the screen. You can adjust the image's contrast by left-click-and-drag on the image or by selecting a region.  

.. |WinLevIcon| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/AdjustWindowLevel.png
   :height: 30

.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ImportMSIFile.png
    :width: 600

Load Pathology Image
---------
#. To load a pathology image, click 'Select file...' under pathology image. A file dialog box will appear, allowing you to upload the desired image. 

.. image:: https://raw.githubusercontent.com/jamzad/SlicerDESI/main/docs/source/Images/LoadPathology.png
    :width: 600

Loading an Existing Project
-----------
If have a saved project you would like to resume working on, click 'Load an existing project..' at the top of the tab. Note that saved projects include annotated scenes, labeled segments, ion visulizations, however, they do not save the MSI file that was used in the previous processing (since the file is too large to store with the proect). After loading a saved project, load the MSI file to continue editing where you left off. 


To clear the scene, navigate to the 'Clear data and start a new project' button at the top of the Data Import Tab. 
