# third party
import os
import numpy as np


def predict_image(stacked_img, model, scaler):
    """Used to predict the stacked image and reshape before and after prediction
    
    Args:
        stacked_img (numpy.ndarray): Stacked image containing 14 layers to preduct
        model (model.fit): A fitted model
    
    Returns:
        numpy.ndarray: predicted bool array of stacked image
    """
    dims = stacked_img.shape
    img_arr = stacked_img.reshape(dims[0], -1).T
    img_arr = scaler.transform(img_arr)
    img_pred = model.predict(img_arr)
    img_pred = img_pred.reshape(dims[1], dims[2]).astype("uint8")

    return img_pred


def proba_image(stacked_img, model, scaler):
    """Used to calculate the mean probability of truck class across each non-masked pixel in the stacked image (only pixels in the road which have not been assigned as cloud). 
    
    Args:
        stacked_img (numpy.ndarray): Stacked image containing 14 layers to predict
        model (model.fit): A fitted model
        scaler (sklearn.scaler): Scaler pre-fitted on training data used to train model
    
    Returns:
        float of mean probability of truck class across the image
    """
    # reshape from 3D to 2D array with features as columns, row for each pixel
    dims = stacked_img.shape
    stacked_img = stacked_img.reshape(dims[0], -1).T

    # Get rid of the masked values -- initially fill the mask with -999
    stacked_img = stacked_img.filled(-999)
    # then using np.all on the row dimension (all column values should be > -999 to be road)
    stacked_img = stacked_img[np.all(stacked_img > -999, axis=1)]

    if stacked_img.shape == (0, 28):
        return 0
    else:
        # Scale the 2d array with the pre-fit scaler
        stacked_img = scaler.transform(stacked_img)
        # image prediction with same nrows as img_arr
        img_pred = model.predict_proba(stacked_img)
        # take the truck class probability
        img_pred = img_pred[:, 1]

        proba_mean = np.mean(img_pred)
        return proba_mean
