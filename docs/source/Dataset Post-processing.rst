Dataset Post-processing
#####
#. In the **Dataset Post-Processing** Tab, locate and click on ‘Select’ to import the converted CSV dataset. 
#. A file dialog box will appear, allowing you to browse and select the desired CSV file from your local storage. Once you have chosen the file, its path will be displayed on the screen, indicating that the file has been successfully selected. 
#. Click ‘Import’. If the file was loaded successfully, a confirmation message will be displayed, stating "Dataset successfully loaded". 

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/60633c05-6df0-452e-ab83-32fcc68fe33a
    :width: 600

Normalization 
---------
To perform normalization on the dataset, check the box beside ‘Normalization’. Total Ion Current (TIC) normalization is automaticcaly applied. If you would like to normalize to a choose reference ion, select 'Refernce Ion' and specify a m/z value from the drop down menu. 

Spectrum Filling
----------
To filter the spectrum and limit the feature space, check the box beside ‘Spectrum Filtering’. In the provided fields, specify m/z values for lower and upper bands of the filtering range.

Pixel Aggregation
--------
To create custom ROI patches by grouping pixels, check the box beside ‘Pixel Aggregation’. This will generate a grid of patches to be overlaid on the segmentation in the dataset, and each patch within the grid becomes an individual sample in the dataset. 
Adjust the settings in the appropriate fields 
    - Patch Width (defines the width of each individual patch in pixels)
    - Stride (defines the space between adjacent patches)
    - Partial Patch Percentage (sets the overlap percentage between neighboring patches)
In the dropdown menu select an aggregation mode to determine how pixels within each patch are combined to represent the patch as a single sample. 


Saving the Processed Dataset 
-----------
When you are done adjusting post-processing settings, click ‘Apply’ at the bottom of the window. Provide a name for the processed dataset and location to save it to on your local storage. 

###will automatically be saved to your working directory as a CSV file named ‘processed_dataset.csv’. ###
