AI Model Deployment 
########
The purpose of this tab is to deploy an AI pipeline analysis on a whole-slide MSI data.

Import MSI data
-------
#. Click 'Select file...' under **Import MSI slide**. A file dialog box will open, allowing you to browse and select the file. The supported data formats are similar to the "Data Import" tab.
#. After the data is loaded, general information like number of pixels, number of ions, and the name and location of the file will appears in the text box. The TIC visualization will also appears on the viewer. 

Import AI model
-------
#. To choose the trained model, click on 'Select file...' under 'Import Model Pipeline' and select the saved pickle file. Click 'Import' to load. 
#. The details of the model will appear in the dialog box.

Preprocessing
-------
#. In 'Preprocessing' section, you can add the option for normalization (TIC or ion-based) pixel aggregation option, or masking (limiting the deployment to specific region on the slide) which will be applied to spectra before going into the model.
#. Users can also limit the model deployment to a specific region of the slide specified by **Mask**. To crate a mask, you should first create a visualization, or select an available one. Click on 'Create mask...' to be redirected to 'Segment Editor' module. You can create a region of interest similar to the ROI selection process manually, or used other available options like thresholding. Use the module arrow kwy on the top of the 3D SLicer to go back to MassVision module, where you can select the generated mask from the dropdown menu.
#. Click on 'Deploy Model' at the bottom of the tab. 