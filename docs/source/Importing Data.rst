Importing Data
=====

To load the data navigate to the **Data Import** tab. You have different options for data format. 

Import MSI Data
-------
#. To import MSI data, click 'Select file...' under MSI data. A file dialog box will open, allowing you to browse and select the file. The current supported formats are DESI image files (.txt) and tabular structured files (.csv). After choosing the file, the path of the selected file will be displayed.
#. After the data is loaded, the generated TIC image will be displayed in the viewer. To adjust the image's contrast, click the |WinLevIcon| icon in the Slicer toolbar at the top of the screen. You can adjust the image's contrast by left-click-and-drag on the image or by selecting a region.  

.. |WinLevIcon| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/AdjustWindowLevel.png
   :height: 30

.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ImportMSIFile.png
    :width: 600

CSV File Structure
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

**Example** 2x3 pixels MSI with 4 ions at m/z of 281.0375, 494.2507, 600.324, 831.5288:
   
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

+---+---+----------+----------+---------+----------+
| 2 | 3 | 281.0375 | 494.2507 | 600.324 | 831.5288 |
+---+---+----------+----------+---------+----------+
| 0 | 0 |    26    |    59    |    9    |    133   |
+---+---+----------+----------+---------+----------+
| 0 | 1 |    45    |    32    |    0    |    24    |
+---+---+----------+----------+---------+----------+
| 0 | 2 |     0    |     0    |    77   |     0    |
+---+---+----------+----------+---------+----------+
| 1 | 0 |    112   |    60    |    18   |    72    |
+---+---+----------+----------+---------+----------+
| 1 | 1 |     0    |    28    |    38   |    22    |
+---+---+----------+----------+---------+----------+
| 1 | 2 |     0    |     0    |    0    |    18    |
+---+---+----------+----------+---------+----------+

By following this format, users can import MSI data from diverse modalities into MassVision for analysis and visualization. 

Import Pathology Image
---------
To load a pathology image, click 'Select file...' under pathology image. A file dialog box will appear, allowing you to upload the desired image. 

.. image:: https://raw.githubusercontent.com/jamzad/SlicerDESI/main/docs/source/Images/LoadPathology.png
    :width: 600

Loading Existing Project
-----------
If have a saved project you would like to resume working on, click 'Load an existing project..' at the top of the tab. Note that saved projects include annotated scenes, labeled segments, ion visualizations, however, they do not save the MSI file that was used in the previous processing (since the file is too large to store with the project). After loading a saved project, load the MSI file to continue editing where you left off. 


To clear the scene, navigate to the 'Clear data and start a new project' button at the top of the Data Import Tab. 

