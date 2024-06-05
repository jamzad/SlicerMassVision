Visualization
#######
Use the **Visualization** tab to visualize the imported MSI data. MassVIsion supports different targeted and non-targeted visualization approaches.


Targeted ion Images
******
Single-ion heatmap
-----------
To view the heatmap of a specific ion, select the m/z value of interest from the 'Ion Image' dropdown. You can type a numerical value to search the dropdown options. Select a heatmap type for the channel and push 'Visualize'.

Multi-ion visualization
-----------
You can also select multiple m/z values in different color channels. Select m/z values for your channels of interest. You can type a numerical value to search the dropdown options and push 'Visualize'.

Untargeted Multivariate
*********
Global Contrast (PCA)
-----------
To visualize data as a PCA image, select 'Global Contrast (PCA)'. The generated image has the first 3 PCs as RGB channels. The PCA is calculated over all the pixels inside th image.

Local Contrast (PCA)
----------
To limit the PCA calculation to a specific spatial location, you should define a region of interest first. Within the markups toolbar appears on top of the module, click on the dice icon to create a new ROI.

.. image :: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/CreateROI.png
    :width: 600

Use the cursor to select an ROI on the histopathology image by creating a bounding box. The dimensions of the box can be adjusted by dragging the dots on the box’s perimeter. Its location can be adjusted by clicking on the center dot and dragging the box. 

If you would like to edit, rename, or delete the bounding box, click the dropdown beside the dice icon and select your desired action. 

.. note::
    If you edit the ROI, you will be directed to the markups module. Navigate back to the MassVision module by clicking the magnifying glass beside 'Modules', and search for MassVision, or use the green arrows beside the 'Modules' to go to previous modules.

| Select 'Local PCA Contrast'. You can move the ROI box to see the contrast underneath. 

| Select 'Extend to whole slide' to apply this contrast to the whole image. 

.. image :: https://raw.githubusercontent.com/jamzad/SlicerMassVision/main/docs/source/Images/ROIonPCA.png
    :width: 600


Pixel Spectrum
******
To visualize a specific spectrum plot for each pixel of your visualization, first click the Markups Toolbar toggle located in the main Slicer Toolbar. The markup toolbar will be displayed below. 
In order to define pixels for spectrum generation, you must add fiducials to the visualization image. Click the three red dots icon (Create new point list). By hovering your cursor over the visualization image you will be able to add a fiducial. You can add as many fiducials as you'd like. To delete or provide a specific name for the control point, right click the point to view options.  
Once you are satisfied with your fiducials, click 'Spectrum Plot'. The viewer will display a plot for each pixel beside the visualization image. 

