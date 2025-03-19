Data Structure
=====

MassVision allows users to import modality independent MSI data in a structured CSV or hierarchical HDF5 formats for analysis. These format ensures compatibility with any rectilinear MSI data where all pixels share a common list of ions. The following instructions assume the MSI data has **MxN pixels** and **L ions** per pixel. 


HDF5 file structure
----------
The HDF5 should have two datasets, one for peak intensities named **peaks**, and one for ions named **mz**. The content of the datasets are the following **Numpy** arrays:

1. **peaks**: A three-dimensional MSI data as Numpy array with shape **(MxNxL)** (float or integer)

2. **mz**: A one-dimensional vector data as Numpy array with shape **(L,)** (float) 


CSV file structure
----------
The CSV should be created based on a spectrum-per-row architecture. The CSV file will have **M×N+1 rows** (one header row and M×N rows for pixel spectra) and **L+2 columns** (two location indices and L abundance values):

1. **Header Row**  
   The first row (header) contains:  
   
   - **M**: Number of pixels in height (integer)  
   - **N**: Number of pixels in width (integer)  
   - **L m/z values**: The m/z values corresponding to L ions (float) 

2. **Pixel Data Rows**  
   Each subsequent row corresponds to a pixel on the M×N grid. Each row contains:  
   
   - **i**: The pixel’s y-coordinate (integer, range 0 to M-1, 0 being the top)  
   - **j**: The pixel’s x-coordinate (integer, range 0 to N-1, 0 being the left)  
   - **Ion intensities**: The abundance values for the L ions at this pixel (no specific type or range constraints) 

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



