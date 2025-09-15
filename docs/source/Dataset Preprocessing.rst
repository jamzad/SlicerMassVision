Dataset Preprocessing
=====================
The **Dataset Preprocessing** tab is designed for spectral and spatial normalization and filtering of the data.


Import 
------
Click on **Select file** under "Import CSV dataset". A dialog box will appear allowing you to browse your local storage and select the desired CSV file. You will be prompted by the successful load message and the information about the dataset including number of slides, number of classes, number of total spectra, and number of spectra per classes will appear in the available text box


Normalization 
-------------
To perform spectrum normalization on the dataset, check the box beside **Normalization**. MassVision currently support 6 spectrum normalization methods:

    - TIC (total ion current) 
    - TSC (total signal current)
    - Reference ion
    - Mean
    - Median
    - RMS


More information on the normalization formulations can be found in `Functions and Formulations <https://slicermassvision.readthedocs.io/en/latest/Functions%20and%20Formulations.html#>`_ page.


Spectrum Filtering
------------------
To limit the m/z range to a specific interval, select the checkbox next to **Spectrum Filtering**. Then, enter the m/z values for the lower and upper bounds of the range in the provided fields.


Pixel Aggregation
-----------------
To aggregate the neighboring spectra in each slide's ROI (spatial denoising), check the box beside **Pixel Aggregation**. The process involves reshaping the data from individual slides in the dataset back to their 2D structure. MassVision then overlay a **regular grid of square patches** with specific patch size and distance on the slide. The available spectra within each patch (if they cover specific percentage of the patch) are summarized to a single spectrum using a defined statistics.

Parameters:
    - Patch width (the width of the neighborhood included in the patch in pixels)
    - Patch distance (the distance between consequent patches on a grid in pixels)
    - Incomplete patch reject (he minimum percentage of available pixels within a patch required for it to be considered complete and included in the aggregation.)
    - Aggregation mode (the statistical operation for aggregating the pixels within the patch)

Apply Processing 
----------------
When you are done adjusting preprocessing settings, click **Apply** at the bottom of the tab. Specify a name and save location for the preprocessed file on your local storage. Once saved, the summary of the processed file will appear in the available text box. Representative visualizations of the patch grids for each slide are also generated and saved in the same location for further exploration.

