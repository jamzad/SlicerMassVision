Supervised AI Model Training 
============================

This tab facilitates feature selection, data partitioning, data balancing, and training supervised methods for tissue type classification.

Dataset Import 
--------------
Click on **Select file** under "Import CSV dataset". A dialog box will appear allowing you to browse your local storage and select the desired CSV file. After loading, a success message will be shown and dataset information will be displayed, including the number of slides, number of classes, total spectra, and spectra per class.

Feature Ranking
---------------
MassVision provides users with multiple feature ranking options. The hyperparameters associated with each method can also be tuned according to the user’s needs. Current methods include:

    - Linear Support Vector Classification
    - Partial Least Squares Discriminant Analysis
    - Linear Discriminant Analysis

Feature Selection
-----------------
Users can control which ions (features) are included in the classification step by choosing from several feature selection options:

    - **None:** Use the complete set of ions in the dataset without restriction.
    - **Top ranked:** Apply one of the available feature ranking algorithms and specify the number of top-ranked ions to retain. This allows classification to focus on the most informative features.
    - **Manual:** Upload a CSV file containing a single column with the indices of hand-picked ions. Only these ions will be used for classification.

Model Training/Validation
-------------------------

Model
*****
Select the AI model from the available list under **Model type** and set the hyperparameters according to your research needs. The available classification models include:
    
    - Principal Component Analysis followed by Linear Discriminant Analysis
    - Linear Support Vector Classification
    - Random Forest
    - Partial Least Squares Discriminant Analysis

Data partitioning
*****************
Choose the data partitioning configuration from **Data split scheme** to determine how data is divided for training and validation. Available options include:
  
    - **Training on whole dataset:** use the entire dataset for training and report performance measures on the training set.
    - **Random train/test split:** randomly divide the data into training and test sets, and report performance measures for both.
    - **Slide-based train/test customization:** manually select which slides/patients are included in the training or test sets using the provided checkboxes. Performance measures are reported for both sets.
    - **Leave-one-slide-out cross-validation:** run the slide-based customization iteratively, leaving one patient/slide out as the test set each time, and report the average performance metrics across folds for both training and test sets.

Data balancing
**************
To mitigate biases from imbalanced training data, MassVision supports three class-based balancing strategies available in the **Data balancing** dropdown:
    
    - **None:** no balancing is applied. 
    - **Undersampling:** randomly exclude spectra from majority classes until each class has the same number of spectra as the minority class.
    - **Oversampling:** randomly replicate spectra from minority classes until each class reaches the number of spectra in the majority class.
    - **Hybrid:** up-sample minority classes and down-sample majority classes to the average number of spectra per class.

Train/validate
**************
Once you are satisfied with the parameters, click **Train and validate** at the bottom of the tab to start training.  
If the **Export model pipeline** box is checked, a dialog will appear prompting you to specify the name and location for saving.  

After training, you will be redirected to the **Performance Report** tab, where you can review details of the training and test data distribution, performance measures, and—if applicable—visualizations such as LDA scatter plots. 

.. important::
   To save the trained classification pipeline for later use on whole-MSI data, be sure to check the box for **Export model pipeline**.
