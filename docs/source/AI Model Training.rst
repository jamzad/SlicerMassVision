AI Model Training 
########

The **AI Model Training** tab is designed for exploration of the data, and training and validation of AI models.

Import 
---------
Click on **Select file** under "Import CSV dataset". A dialog box will appear allowing you to browse your local storage and select the desired CSV file. You will be prompted by the successful load message and the information about the dataset including number of slides, number of classes, number of total spectra, and number of spectra per classes will appear in the available text box

Data Distribution
-------
To view the scatter plot of spectra in lower dimension, click on **Plot PCA latent space** under "Data Distribution". The scatter plot is color coded both for slides and class label Differentiation. The generated image of scatter plots will also be saved to the same path as the imported CSV dataset. 

Model Training
-------
#. Select the AI model from the available list (PCA-LDA, SVM, Random Forest, PLS-DA) under **Model type**.
#. Choose the data partitioning configuration from **Data split scheme** to determine the division of data for train and validation.

    Options:    
        - Training on whole dataset: use the whole dataset for training and report the training performance measures.
        - Random train/test split: divide the data randomly into train/test and report the performance measures for both train and test set.
        - Slide-based train/test customization: user can choose which slides/patients from the dataset included in train or test by selecting the appropriate checkboxes from the provided list. The performance report would be for both train and test set.

#. To balance the training data and equalize the number of spectra in each class, select one of the available options from **Data balancing**.
    
    Options:
        - None: no balancing
        - Down-sample: randomly exclude spectra from classes to have the same number of spectra as the the minority class.
        - Up-sample: randomly repeat spectra within each class to reach the number of spectra in the majority class
        - Mid-sample: up-sample the classes with low number of spectra, while down-sample the classes with high number of spectra, to the average spectrum per class value

#. if you want to save the model for later use on the whole slide, please check the box for **Export model pipeline**.

#. When you are satisfied with the parameters, click on **Train and validate** at the bottom of the tab to start the training. If the model export box is checked, a dialog will appear for thr user to specify the name and the location for saving. You will then be taken to the **Performance Report** tab to review the details of the train and test data distribution, along with performance measures for classification, and additional visualizations like LDA scatter plots. 


