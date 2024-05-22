Generating a Dataset
######
To generate a suitable dataset, the histopathology and PCA images must undergo preprocessing and registration prior to segmentation. 
 
Visualize the histopathology and PCA images together
*******
#. Return to the **Data Import** Tab. Load the histopathology image again; it will appear on the right side of the split viewer. 
#. Click on the pin icon located in the top left corner of the red band above the left side of the split screen. In the dropdown toolbar, change the current volume to the PCA image. 
#. Ensure the viewer for both images is set to ‘Axial’ so they are correctly correlated together. If needed, click the grid icons in the top bands to center the images according to the field of view. 

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/f3a1be3c-1e2a-4049-b6ae-87bb41def7ac
   :width: 600

Registration
*******
#. Navigate to the **Dataset Generation** tab. Beside 'Histopathology colocalization', click on the 'Landmark Registration'. This will switch to the Landmark Regtistration module. 
#. In the ensuing dialog box, designate the PCA volume as the ‘Fixed Volume’, and the pathology image as the ‘Moving Volume’. Click ‘Apply’. |ico1|
#. The PCA, pathology and transformed images will all be displayed on the left side of the screen. Beside ‘Zoom’, click ‘Fit’ to center all the images.

.. |ico1| image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/73fba211-ec39-47e5-bb31-15c7f8d97498
   :width: 150

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/8c9aef61-b5f4-495c-b604-8fc67d086186
   :width: 600

Landmarking
*******
.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/a8cb6a37-26e9-4a54-8828-03b435c8e02d
   :width: 600

#. Set desired parameters for the landmark registration. For the purpose of this this tutorial, select ‘Affine Registration’ for the ‘Registration Type’, and ‘Similarity’ for the ‘Linear Regression Registration Mode’.
.. note::
   For more information about what each registration parameter means please visit: `<https://www.slicer.org/wiki/Documentation/Nightly/Modules/LandmarkRegistration>`_
#. Click 'Add Landmark' and add a fiducial landmark to the PCA image.  
#. After selecting a landmark on the PCA image, a corresponding dot will appear on the histopathology image in the anticipated corresponding location. Drag the dot on the pathology image to the correct location. If necessary, re-fit the images with the grid icon.
#. Continue this process for 3-4 landmarks to ensure proper alignment and registration of the images. 
#. Upon completion, navigate to the green viewer at the bottom of the display select the maximize view button to see the histopathology/PCA overlay in the entire viewer.

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/4c2d2b7a-a2d7-4118-b4ff-4d4d4fa3e891
   :width: 600

ROI Selection
********
#. Return to the previous module (SlicerDESI) by clicking green back arrow in the toolbar. Alternatively, you can switch between modules by clicking the magnifying glass in the toolbar beside 'Modules' and searching for your desired module. 
#. In the module’s **Dataset Generation** tab, next to ‘ROI selection’, click on 'Update Visulization Lists' to see options. You can select one image for ROI selection, and another as a guide. For the purpose of this tutorial, select a PCA Visulization for the Main visulization for ROI selection, and the previously generated registered histo image for the Guide Visulization for ROI selection. 
#. Click on ‘Segment editor’. This will bring you to the Segment editor module, where the two choosen images will be displayed in the side-by-side split screen viewer. 
#. Once open, ensure that the segmentation name is ‘Segmentation’. ####Change the source volume to match the volume of the PCA image and#### .Select the dice icon beside the master volume. Change the source geometry to the PCA image as well and click ‘OK’.
#. To add segments, click ‘Add’. This action will create a new segment which you can name as per your preference.  
#. To draw on the image, select the second from the top leftmost icon (just below the mouse button) |ico2|
#. When you have finished drawing your desired segment, click ‘Add’ again to start the next segment. Add at least three segments.
#. Once you’re satisfied with your image, head to the module header in Slicer to return to the SlicerDESI module. Click the green back arrow in the top toolbar to navigate to the previous module. 
#. Within the **Dataset Generation** tab of the module, select ‘Create .csv dataset’ next to ‘Dataset Generation’. You will be prompted to enter a file name and location on your local computer before saving the generated segmentations as a CSV file. 
#. Additionally, you have the option to generate images for your segments. Once again, create a name and specify the location you would like to save to, the segments that you generated will be saved as mrb files. ####as images to your working directory (same directory as where your PCA image was created and the data was loaded in from) allowing you to view them at a later time.#### 

.. |ico2| image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/61f6b345-0ee4-4b07-af19-848eaf6fc9d4

Saving Your Project and Generating More Datasets 
********
If you would like to analyze another single slide .. To save your project, click on 'Save ROIs and visulizations..' at the bottom of the tab. Your scene will be saved as a mrb file. 

.. note::
   When saving your project, make sure the file name is not too long, as long mrb file names will generate an error and not save. 

.. note:: 
   When saving your scene and reopening it an another ocasion, you **must** import your raw DESI dataset in order to resume your visulization, dataset processing and generation. Your previously recorded scene does not retain this data. To accomplish this, simply go to **the Data Import** tab and import the equivalent data.