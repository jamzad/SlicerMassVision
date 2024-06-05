Generating a Dataset
######
To generate a suitable dataset, the histopathology and MSI visualization must undergo preprocessing and registration prior to ROI selection. Use **Dataset generation** tab for this task
 
Co-localization
*******
#. Beside 'Histopathology colocalization', click on the 'Landmark Registration'. This will switch to the Landmark Registration module in 3D Slicer. 
#. In the ensuing dialog box, designate an MSI visualization, preferably PCA, as the ‘Fixed Volume’, and the pathology image as the ‘Moving Volume’, then click ‘Apply’.
#. The PCA, pathology and overlayed transformed images will all be displayed on the left side of the screen. Beside ‘Zoom’, click ‘Fit’ to center all the images.

   .. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/RegistrationMenu.png
       :width: 600

#. Set desired parameters for the landmark registration. For the purpose of this this tutorial, select ‘Affine Registration’ for the ‘Registration Type’, and ‘Similarity’ for the ‘Linear Regression Registration Mode’.

.. note::
For more information about what each registration parameter means please visit: `<https://www.slicer.org/wiki/Documentation/Nightly/Modules/LandmarkRegistration>`_

#. Click 'Add Landmark' and add a fiducial landmark to the PCA image.  
#. After selecting a landmark on the PCA image, a corresponding dot will appear on the histopathology image in the anticipated corresponding location. Drag the dot on the pathology image to the correct location. If necessary, re-fit the images with the grid icon.
#. Continue this process for 3-4 landmarks to ensure proper alignment and registration of the images. 
#. Upon completion, navigate to the green viewer at the bottom of the display select the maximize view button to see the histopathology/PCA overlay in the entire viewer.


ROI Selection
********
#. Return to the MassVision module by clicking green back arrow in the toolbar. Alternatively, you can switch between modules by clicking the magnifying glass in the toolbar beside 'Modules' and searching for your desired module. 
#. In the module’s **Dataset Generation** tab, next to ‘ROI selection’, you can select two visualizations; one as the Main reference and the other as the guide. For the purpose of this tutorial, select a PCA Visualization for the reference visualization for ROI selection, and the previously generated registered histo image for the Guide Visualization for ROI selection. 
#. Click on ‘Segment editor’. This will bring you to the Segment editor module, where the two chosen images will be displayed in the side-by-side split screen viewer. 
#. To add segments, click ‘Add’. This action will create a new segment which you can name as per your preference.  
#. To draw on the image, select the second from the top leftmost icon (just below the mouse button)
#. When you have finished drawing your desired segment, click ‘Add’ again to start the next segment. Add at least three segments.
#. Once you’re satisfied with your image, head to the module header in Slicer to return to the MassVision module. Click the green back arrow in the top toolbar to navigate to the previous module. 
#. Within the **Dataset Generation** tab of the module, select ‘Create .csv dataset’ next to ‘Dataset Generation’. You will be prompted to enter a file name and location on your local computer before saving the generated segmentation as a CSV file. 
#. Additionally, you have the option to generate images for your segments. Once again, create a name and specify the location you would like to save to, the segments that you generated will be saved as mrb files. ####as images to your working directory (same directory as where your PCA image was created and the data was loaded in from) allowing you to view them at a later time.#### 


Saving Your Project and Generating More Datasets 
********
If you would like to analyze another single slide .. To save your project, click on 'Save ROIs and visualizations..' at the bottom of the tab. Your scene will be saved as a mrb file. 

.. note::
   When saving your project, make sure the file name is not too long, as long mrb file names will generate an error and not save. 

.. note:: 
   When saving your scene and reopening it an another occasion, you **must** import your MSI data in order to resume your visualization, dataset processing and generation. Your previously recorded scene does not retain this data. To accomplish this, simply go to **the Data Import** tab and import the equivalent data.
