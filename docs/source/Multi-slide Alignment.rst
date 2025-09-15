Multi-Slide Alignment 
=====================

When working with MSI datasets from multiple slides (e.g., different patients or tissue sections), each slide typically contains a unique m/z list. This prevents direct comparison or merging of datasets. The **Multi-Slide Alignment** tab in MassVision provides tools to unify m/z features across slides before exporting a merged dataset for downstream analysis.

Import datasets
---------------

- Import as many CSV files as desired by clicking **Select** under "Import CSV datasets" section. A dialog box will appear allowing you to browse your local storage and select files. 
- After adding all data, you see the names and locations in list. You can add more files by clicking the **Select** button again, or delete existing files by clicking on **Delete** in front of each item.
- After satisfied with the list, click on **Load files** to load the data.

.. important::
    Differentiation between slides/patients is based on the names of the individual CSV datasets. Therefore, each dataset name must be unique to ensure error-free slide tracking.

Alignment Parameters
--------------------

**KDE bandwidth for ion clustering**  
    Controls the standard deviation of the Gaussian kernel used in Kernel Density Estimation (KDE). This parameter defines the radius (in Da) for grouping neighboring m/z values into clusters. 

    - Smaller values → narrow clustering, preserving fine details but requiring higher mass accuracy.  
    - Larger values → broader clustering, merging nearby peaks more aggressively.

**Feature sparsity**  
    Specifies the criterion for retaining clusters (features) after alignment.  
    For example, a sparsity value of ``0.7`` means that a feature will be discarded if it appears in **30% or fewer spectra or slides**, depending on the selected **Sparsity level**.  

    - Higher sparsity values → keep only features shared broadly across the dataset, reducing the number of total features.  
    - Lower sparsity values → retain more features, including those that may be rare or slide-specific.  

**Sparsity level**  
    Defines whether sparsity is calculated based on **Slides** or **Spectra**. 

    - *Spectra*: the threshold is applied across individual spectra within slides.  
    - *Slides*: the threshold is applied across entire slides, retaining only features consistently present between slides.  

Alignment Preview
-----------------

**visualization range (m/z)**  
    Enter an m/z window to inspect alignment quality in a specific region. 

**Visualize**  
    Generates the interactive KDE density preview, helping confirm that peaks from different slides are correctly grouped. The preview shows: 

    - The original m/z values,  
    - KDE-derived density peaks (clusters), and  
    - The final selected feature list after applying sparsity.  

Peak Matching
-------------

**Method**  
    Select how intensities are assigned to the unified feature list:  

    - *Cluster*: assigns each peak to the nearest KDE-derived cluster.  
    - *Tolerance mode*: assigns peaks within a user-defined m/z window around each feature.  

    In this mode, multiple matches can be aggregated using statistics such as **mean**, **sum**, or **maximum**.  

**Align and Save**  
    Performs alignment using the selected parameters and saves the merged dataset in CSV format in the user defined location. The aligned spectra from all slides are mapped to a single unified m/z reference list. The metadata about the aligned dataset detailing the number of slides, classes, and spectra per classes will be displayed after alignment.
