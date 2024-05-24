Importing Data
=====

The first thing that you will want to do when using the module is to load in the required data. 
Navigate to the **Data Import** tab. 

Import DESI Data
-------
#. To import the DESI data, click 'Select'. A file dialog box will open, allowing you to browse and select the text file containing the DESI data from your local storage. After choosing the file, the path of the selected file will be displayed on the screen, confirming that the file is ready for import.
#. Click 'Import'. If the file is loaded successfully, a confirmation message will appear, stating "Slide successfully loaded." Additional details about the imported file will also be provided. 
#. The geneerated TIC image will be displayed in the viewer. To adjust the image's contrast, click the .. icon in the Slicer toolbar at the top of the screen. You can adjust the image's contrast by left-click-and-drag on the image or by selecting a region.  

.. image:: https://raw.githubusercontent.com/jamzad/SlicerDESI/main/docs/source/Images/ImportMSIFile.png
    :width: 600
    :align: center

Load Pathology Image
---------
#. To load a pathology image, click 'Select', a file dialog box will appear, allowing you to upload the desired image. 
#. Click 'Load'. After the image is uploaded, it will be displayed in viewer within the module's interface.

.. image:: https://raw.githubusercontent.com/jamzad/SlicerDESI/main/docs/source/Images/LoadPathology.png
    :width: 600
    :align: center

Import REIMS CSV data
-------
#. [Inset instructions]


Loading an Existing Project
-----------
If have a saved project you would like to resume working on, click 'Load an existing project..' at the top of the tab. Note that saved projects include annotated scenes, labeled segments, ion visulizations, however, they do not save the DESI file that was used in the previous processing (since the file is too large to store with the proect). After loading a saved project, load the DESI file to continue editing where you left off. 



To clear the scene, navigate to the 'Clear data and start a new project' button at the top of the Data Import Tab. 
