# fcdo_trucks_project
FCDO Trucks ML Project from ONS Data Science Campus, for code review.

This mini-project is a case study of the ["Detecting Trucks in East Africa"](https://datasciencecampus.ons.gov.uk/detecting-trucks-in-east-africa/) project done by the ONS and the FCDO.

The aim of this project was to understand more about imbalanced learning and model explanation, through scikit-learn.

The notebook and scripts in the `ONS` folder have been produced by the ONS. The notebook `Trucks_modelling.ipynb` contains some exploratory data analysis, and training of an RFC model, followed by k-fold cross validation. Finally, they apply the model to the 'image chip' data (small portions of images), to estimate the number of trucks on each day along the particular section of road. The scripts in the `ONS/truck_detection` folder contain scripts to aid with this (loading in image chips, scaling them, and applying the model).

For this mini-project, I have written the following notebooks:

 1. `1_exploratory_data_analysis.ipynb` Explaining the data included in the mini-project and explore some of the features.
 2. `2_feature_selection.ipynb` Exploring feature selection methods included in scikit-learn.
 3. `3_model_selection.ipynb` Comparing models included with scikit-learn, exploring the use of the `imblearn` package to help with training from highly imbalanced data, and the use of `shap` to help interpret the model.