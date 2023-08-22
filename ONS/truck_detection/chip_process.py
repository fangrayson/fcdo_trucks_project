## THIRD PARTY
import numpy as np
import pandas as pd

## PROJECT
from .truck_detection import *


def classify_chip_nogeos(location, img_date, model, scaler):
    """Classify image chips using pre-trained model and output shapefile of results
    
        Args:
            model (sklearn.classifier): Pre-trained model to use for classification
            scaler (sklearn.scaler): Scaler pre-fitted on training data used to train model
            output_results_fp (string): Full path + file name for results dataframe
        
        Returns:
            pandas.DataFrame of classified results"""
    
    stacked_arrays = [
    os.path.join("../data/chips", x)
    for x in os.listdir("../data/chips")
    if all(keyword in x for keyword in [location, img_date, "__"])
    ]
    
    proba_chips = []
    
    count_chips = []
    
    for array in stacked_arrays:
        
        loaded_arr = np.load(array, allow_pickle=True)
        
        #calculate mean probability of positive class (truck) across image
        mean_truck_proba = proba_image(loaded_arr, model, scaler)
        proba_chips.append(mean_truck_proba)
        
        # caluclate count prediction across image 
        class_res = predict_image(loaded_arr, model, scaler)
        
        if class_res.max() > 0:
            count_chips.append(class_res.sum())
 
    # mean probability across all image chips
    mean_of_chips = np.mean(proba_chips)
    
    # sum of predicted truck counts across all image chips
    sum_of_chips = sum(count_chips)
    
    return sum_of_chips, mean_of_chips
