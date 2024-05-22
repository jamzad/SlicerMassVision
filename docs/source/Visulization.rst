Visulization
#######
In the **Visualization** tab, DESI data can be visualized in the following ways.

Pixel Spectrum
******
To visualize a specific spectrum plot for each pixel of your visulization, first click the Markups Toolbar toggle located in the main Slicer Toolbar. The markup toolbar will be displayed below. 
In order to define pixels for spectrum generation, you must add fiducials to the visulzation image. Click the three red dots icon (Create new point list). By hovering your cursor over the visulization image you will be able to add a fiducial. You can add as many fiducials as you'd like. To delete or provide a specific name for the control point, right click the point to view options.  
Once you are satisfied with your fiducials, click 'Spectrum Plot'. The viewer will display a plot for each pixel beside the visulization image. 

Targeted Single-ion
******
If you wish to view as a single ion image, select one of the m/z values from the 'Ion Image' dropdown. You can type a numrical value to search the dropdown options.  
Select a heatmap for the channel. Select 'Visualize'. In the display view, the pathology image will be displayed. 

Targeted Multi-ion
********
You can also select multiple m/z values in different colour channels. Select m/z values for your channels of interest. You can type a numrical value to search the dropdown options. Select 'Visualize'. In the display view, the pathology image will be displayed. 

Untargeted Multivariate
*********
Global Contrast (PCA)
-----------
To visualize data as a PCA image, select 'Whole Slide PCA contrast'. In the display view, the histopathology image will automatically align to the center.

Local Contrast (PCA)
----------
| To define a ROI for local PCA contrast, locate the toolbar at the top of the screen and click on the Toggle Markups Toolbar icon |ico1| The markups toolbar will appear underneath the main toolbar. Within the markups toolbar, click on the dice icon to create a new ROI. Use the cursor to select an ROI on the histopathology image by creating a bounding box. The dimensions of the box can be adjusted by dragging the dots on the box’s perimeter. Its location can be adjusted by clicking on the center dot and dragging the box. 
| Toggle Markups Toolbar icon: 

.. |ico| image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/24268164-3679-42b7-adc5-e33e748f4176

| Within the markups toolbar, click on the dice icon to create a new ROI. Use your cursor to draw a bounding box on the image. Adjust its dimensions by dragging the dots on the box’s perimeter, and reposition it by clicking and dragging the center dot. 

.. image :: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/6803cc96-039d-4a5a-8460-204b635bd158
    :width: 600

| If you would like to edit, rename, or delete the bounding box, click the dropdown beside the dice icon and select your desired action. 

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/2baa32c3-38c9-4f96-b06f-14640b618de5
    :width: 600

.. note::
    If you edit the ROI, you will be directed to the markups module. Navigate back to the ViPRE module by clicking the magnifying glass beside 'Modules', and searching for ViPRE.
|
| Select 'Local PCA Contrast'. You can move the ROI box to see the contrast underneath. 

.. image :: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/45fb2681-3936-4a4b-a423-52bbd44737e0
    :width: 600

| Select 'Extend to whole slide' to apply this contrast to the whole image. 



