Model Training 
########
#. To proceed with model training, navigate to the **Model Training** Tab.
#. Upload the processed CSV file by clicking ‘Select’ and importing it from your local storage. Once the file is chosen, the file path will be displayed. 
#. Click ‘Import’. If the file is successfully loaded, a message ‘Dataset successfully loaded’ will appear along with information about the file. 

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/daa61960-fb00-4e88-ba65-9f14352dbb34
    :width: 600

Data Distribution
-------
To view the data distribution in latent space, click 'Plot PCA latent space'. The The plotted slide distribtuion and class distribution will appear in the viewer. The generated image will aslo be saved to the location of the CSV file that you are working with. 

Model Training
-------
Among the four implemented models available (PCA-LDA, SVM, Random Forest, PLS-DA), choose the model that you wish to run by selecting it from the dropdown menu. 

To partition data between test and training sets, choose a split from the options provided. 
If you choose a custom split, allocate files to training or testing in the popup that appear at the bottom of the window. 

When you are done adjusting training settings, click ‘Train model’ at the bottom of the window. 

.. note:: 
    To save the model as a pkl file to be used later on in deploymenet, click the box next 'Export pipeline for deployment' before training. The pipeline will be saved in the directory of     the CSV file you are working with, unless specified otherwise. 

You will be taken to the **Performance Report** tab. If your selected model produces a visulization output (such as LDA scatterplot), it will be displayed in the viewer. Information about training, data distribution and performance will be reported for your observance.  

.. image:: https://github.com/jadewarren/ViPRE2.0-clone/assets/132283032/5b7c013d-6364-48ad-a2db-8ea33df5e39f
    :width: 600
