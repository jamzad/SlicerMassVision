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
#. Click on 'Deploy Model' at the bottom of the tab. 