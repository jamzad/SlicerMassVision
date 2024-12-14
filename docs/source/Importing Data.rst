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
#. Click 'Select file...' under MSI data. A file dialog box will open, allowing you to browse and select the file. The current supported formats are DESI image files (.txt) and general MSI structured files (.csv). Click 'Open' to load the data.
#. After the data is loaded, general information like number of pixels, number of ions, and the name and location of the file will appears in the text box. The TIC visualization will also be displayed in the viewer. To adjust the image's contrast, click the |WinLevIcon| icon in the Slicer toolbar at the top of the screen. You can adjust the contrast by left-click-and-drag on the image or by selecting a region.  

.. |WinLevIcon| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/AdjustWindowLevel.png
   :height: 30

.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ImportMSIFile.png
    :width: 600

How to structure your CSV
#######
MassVision allows users to import MSI data in a structured CSV format for analysis. This format ensures compatibility with any MSI data where all pixels share a common list of ions. For a data with **MxN pixels** and **L ions** per pixel, the CSV file should have **M×N+1 rows** (one header row and M×N rows for pixel spectra) and **L+2 columns** (two location indices and L abundance values):

1. **Header Row**  
   The first row (header) contains:  
   
   - **M**: Number of pixels in height (integer)  
   - **N**: Number of pixels in width (integer)  
   - **L m/z values**: The m/z values corresponding to L ions (float) 

2. **Pixel Data Rows**  
   Each subsequent row corresponds to a pixel on the M×N grid. Each row contains:  
   
   - **i**: The pixel’s y-coordinate (integer, range 0 to M-1, 0 being the top)  
   - **j**: The pixel’s x-coordinate (integer, range 0 to N-1, 0 being the left)  
   - **Ion intensities**: The abundance values for the L ions at this pixel. (no specific type or range constraints) 

**Example** For a 3x2 pixel MSI data the spatial indexing of the pixels will look like

+-----+-----+
|(0,0)|(0,1)|
+-----+-----+
|(1,0)|(1,1)|
+-----+-----+
|(2,0)|(2,1)|
+-----+-----+

 ..
     .. code-block:: csv

      2, 3, 281.0375, 494.2507, 600.324, 831.5288
  
 

   **Example Pixel Data** for a 2×3 grid with 4 ions:  
   
   .. code-block:: csv

      0,0,26,59,9,133
      0,1,45,32,0,24
      0,2,0,0,77,0
      1,0,112,60,18,72
      1,1,0,28,38,22
      1,2,0,0,0,18
Assume the data contains 4 ions with m/z of 281.0375, 494.2507, 600.324, and 831.5288, the CSV structure will look like

+---+---+----------+----------+----------+----------+
| 3 | 2 | 281.0375 | 494.2507 | 600.324  | 831.5288 |
+===+===+==========+==========+==========+==========+
| 0 | 0 |    26    |    59    |    9     |    133   |
+---+---+----------+----------+----------+----------+
| 0 | 1 |    45    |    32    |    0     |    24    |
+---+---+----------+----------+----------+----------+
| 1 | 0 |     0    |     0    |    77    |     0    |
+---+---+----------+----------+----------+----------+
| 1 | 1 |    112   |    60    |    18    |    72    |
+---+---+----------+----------+----------+----------+
| 2 | 0 |     0    |    28    |    38    |    22    |
+---+---+----------+----------+----------+----------+
| 2 | 1 |     0    |     0    |    0     |    18    |
+---+---+----------+----------+----------+----------+

By following this format, users can import MSI data from diverse modalities into MassVision for analysis and visualization. 

Import Reference Image
---------
To load a gold-standard image like histopathology annotations, click 'Select file...' under Reference Image. A file dialog box will appear, allowing you to upload the desired image. 

.. image:: https://raw.githubusercontent.com/jamzad/SlicerDESI/main/docs/source/Images/LoadPathology.png
    :width: 600



