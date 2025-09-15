Supervised AI Model Deployment 
==============================
This tab is used to deploy an AI pipeline analysis on whole-slide MSI data.

Import Data
-----------
Click **Select file...** under **Import MSI slide**. A file dialog box will open, allowing you to browse and select the file. The supported data formats are the same as those described in the `Data Import/Export <https://slicermassvision.readthedocs.io/en/latest/Data%20Import.html>`_ section.

After the data is loaded, general information such as the number of pixels, number of ions, and the name and location of the file will appear in the text box. A TIC visualization will also appear in the viewer. 

Import Model
------------
To import a trained model, click **Select file...** under **Import Model Pipeline** and select the saved pickle file. Then click **Import** to load it. 

After a successful load, details of the model—including model type, dataset, data split configuration, number and names of the classes, and the number and range of *m/z* features—will appear in the dialog box.

Preprocessing
-------------
In the **Preprocessing** section, users may apply data normalization, pixel aggregation, or spatial masking prior to classification. 

Normalization
*************
The available normalization approaches are the same as those described in the `Normalization <https://slicermassvision.readthedocs.io/en/latest/Dataset%20Preprocessing.html#normalization>`_ section of Dataset Preprocessing, with detailed formulations provided on the `Functions and Formulations <https://slicermassvision.readthedocs.io/en/latest/Functions%20and%20Formulations.html>`_ page.

    .. important::
        Please ensure that the normalization method matches the one used during dataset preparation for model training. Using a different normalization may lead to inconsistent or inaccurate results.

Pixel aggregation
*****************
The pixel aggregation methods are the same as those described in the `Pixel Aggregation <https://slicermassvision.readthedocs.io/en/latest/Dataset%20Preprocessing.html#pixel-aggregation>`_ section of Dataset Preprocessing, with detailed formulations provided on the `Functions and Formulations <https://slicermassvision.readthedocs.io/en/latest/Functions%20and%20Formulations.html>`_ page.

    .. note::
        Unlike normalization, pixel aggregation during deployment does not need to match the settings used in dataset preparation for training. Users may adjust the aggregation level according to their requirements for spatial filtering.

Spatial masking
***************
Users may limit model deployment to a specific region of the slide defined by a **Mask**. 

- To create a mask, first generate or select a visualization.  
- Click **Create mask...** to be redirected to the **Segment Editor** module.  
- Define a region of interest manually (similar to `ROI Selection <https://slicermassvision.readthedocs.io/en/latest/Dataset%20Curation.html#roi-selection>`_), or use other available 3D Slicer tools such as thresholding.  
- Use the module navigation arrow at the top of 3D Slicer to return to the MassVision module, where you can select the generated mask from the dropdown menu.  

Deploy model
------------
When satisfied with the settings, click the **Deploy Model** button to apply preprocessing and classification to the masked pixels of the imported MSI slide. The result will appear as a color-coded image, where each pixel is assigned a color corresponding to the predicted class from the training labels.

