Visualization
#######
Use the **Visualization** tab to visualize the imported MSI data. MassVIsion supports different targeted and untargeted visualization approaches. 

Targeted Visualizations
******
Single-ion heatmap
-----------
Under 'Targeted single-ion' select the m/z value of interest from the 'Ion Image' dropdown, or type the m/z value while on the list to find and select the ion. Select a heatmap option and push 'Visualize' to see the single-ion image on the viewer.

Most abundant ions
-----------
In 'Targeted single-ion' section, click on **Thumbnail view** to see a grid image of TIC and top ion images with highest overall intensity across the slide. 

.. image :: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/Abundant.jpeg
    :width: 600

Multi-ion visualization
-----------
Under 'Targeted multi-ion' select multiple m/z values associated with different color channels using a similar instruction for single-ion. MassVision supports overlaying up to 7 ion images using distinct color scales. Push 'Visualize' to see the result in the viewer.

Untargeted Multivariate
*********
These approaches leverage all ions and apply Principal Component Analysis (PCA) to generate color representations of the MSI data. The visualization is created by concatenating the first three principal components as the red, green, and blue channels of an RGB image. 

Global-contrast
-----------
Select 'Global Contrast (PCA)' to create a PCA visualization based on all the pixels in the MSI data.

Local-contrast
----------
Use this visualization to emphasize the local metabolomic changes by recalculating the PCA specifically for a spatial region of interest (ROI) defined by the user. 

Within the markups toolbar appears on top of the module, click on the dice icon to activate rectangular region selection

.. image :: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/CreateROI.png
    :width: 600

Use the mouse cursor on the PCA image and create a bounding box by left-click-and-drag. The dimensions of the box can be adjusted by dragging the dots on the box’s perimeter. Its location can be adjusted by clicking on the center dot and dragging the box. 

If you would like to edit, rename, or delete the bounding box, click the dropdown beside the dice icon and select your desired action. 

.. note::
    If you edit the ROI, you will be directed to the Markups module. Navigate back to the MassVision module by clicking the magnifying glass beside 'Modules', and search for MassVision, or use the green arrows beside the 'Modules' to go to previous modules.


After the region is specified, click on 'Local Contrast (PCA)' to see generate the visualization for the region. To apply the calculated local PCA to the whole image, check 'Extend to whole slide'. 

.. image:: https://www.dropbox.com/scl/fi/tiwy6mm8pompeeprexf0l/visualization.gif?rlkey=tqtly8rqeymvxkhmdf9hf4039&st=anz46hn1&raw=1
    :width: 600

..
    .. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ROIonPCA.png
        :width: 600

.. tip::
    For all the visualizations, users can adjust the brightness and contrast of the image by selecting the 'Adjust Window/Level' icon |WinLevIcon| in the 3D Slicer toolbar at the top of the screen. You can adjust the contrast by left-click-and-drag on the image or by selecting a region on it.  

.. |WinLevIcon| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/AdjustWindowLevel.png
   :height: 30

Contrast dominant ions
----------
Under 'Untargeted multivariate' section click on **Thumbnail view** to see a grid image of individual PC images and top ion images with highest positive and negative loadings. These ion images are considered to be most dominant in creating the final untargeted visualization. The thumbnail view works for all global, local, and ROI contrast images.

.. image :: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/Loadings.png
    :width: 600

Contrast clustering
******
After generating local or global contrast images, users can run **K-means clustering** on these visualizations to cluster the pixels. Users can compare each cluster with ion images and find ions with highest spatial co-localization with the cluster. This is useful for spatially validated ion identification.

Clustering
----------
After generating your untargeted visualization, select the number of clusters using the dropdown menu and click **Cluster**. The view will display the clustering results as a label map image, where pixels belonging to the same cluster are assigned the same color, distinct from the colors of pixels in other clusters.

Cluster dominant ions
----------
After clustering, select a specific cluster from the dropdown menu, which is color-coded to match the cluster image for convenience. Click on **Thumbnail view** to display a grid image of selected cluster and highly correlated ion images with that cluster. While the ranking is based on **Pearson corelation coefficient**, other measures like **fold change**, **p-value of T-test**, and **Dice score** between the cluster and highly correlated ion images are also reported in a table under 'Cluster dominant ions'.

.. image :: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/Cluster.png
    :width: 600

Pixel Spectrum
******

To plot the mass spectra associated with specific pixels, users should first identify the pixels of interest by placing 'fiducials' at their locations on the viewer. Any visualization can be used for guiding the mass spectrum plot.

#. Click the "Create New Point List" icon |PointList| on the Markups Toolbar.
#. Click on the desired location on the image to place a fiducial (marker).
#. To add another fiducial, select the "Place a Control Point" icon |PlacePoint| and click on the desired location within the visualization.
#. There is no limit to the number of fiducials, allowing you to add as many as needed.
#. To delete or rename a fiducial, right-click on it to view available options.
#. Once you are satisfied with the list of markers, click "Spectrum Plot". A second viewer will display a plot for each selected pixel next to the visualization image.

.. image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/PlotSpectra.png
    :width: 600

.. |PointList| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/PointList.png
   :height: 30

.. |PlacePoint| image:: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/PlacePoint.png
   :height: 30