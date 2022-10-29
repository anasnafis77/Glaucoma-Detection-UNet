import cv2 
import sys
import time
import os
import math
import pickle
import numpy as np
import tkinter as tk
from skimage.transform import resize
from skimage.segmentation import slic
from matplotlib import pyplot as plt
from keras import backend as K
from tkinter import filedialog
from localization import OD_localization
from segmentation import optic_segmentation
from classification import inference_glaucoma
from supporting_function import *

if __name__ == "__main__":
    # Load module objects
    ODFinder = OD_localization() # localization model
    segModel = optic_segmentation() # segmentation model
    glPredictor = inference_glaucoma() # glaucoma predictor

    # Define Hyperparameter 
    ROI_SIZE = 550
    R_COEFF, G_COEFF, B_COEFF, BR_COEFF = 1, 0.2, 0, 0.8 # grid search result
    coeff_args = (R_COEFF, G_COEFF, B_COEFF, BR_COEFF)
    STD_SIZE = (2000, 2000)

    # load image and ROI 
    ret_img = cv2.imread(file_path, 1)
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)

    # preprocessing
    ret_img = ODFinder.preprocessing(ret_img, STD_SIZE)

    # Localize Optic Disc
    start_loc = time.time()
    disc_center = ODFinder(ret_img, coeff_args=coeff_args)
    end_loc = time.time()

    # OD and OC segmentation 
    start_seg = time.time()
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(retinal_image[:, :, 1])
    ROI, coordinate = ekstrakROI(ret_img, ROI_size, cl_img)
    OD_pred, OC_pred = segModel.do_segmentation(ROI, coordinate, )
    end_seg = time.time()


    print('Localization time: {:.2f} s'.format(end_loc-start_loc))
    print('Segmentation time: {:.2f} s'.format(end_seg-start_seg))

    # Inference the feature
    VCDR,HCDR,ACDR = CDR_calc(OD_mask, OC_mask)
    feature = np.array([VCDR, ACDR]) # best prediction achieved by using only VCDR and ACDR
    pred = glaucoma_predict(inference_model, feature)

    print('Detection report of file ', filename)
    print('Prediction: {}'.format(pred))

    fig, ax = plt.subplots(1, 1,  figsize= (5, 5))
    ax.imshow(retinal_image)
    ax.contour(OC_pred, colors='b')
    ax.contour(OD_pred, colors='w')
    ax.grid(False)
    ax.set(title='OD/OC segmentation')
    ax.text(200, 200, 'Prediction: {}'.format(pred), fontsize='medium', color ='w')
    plt.draw()
    while True:
        if plt.waitforbuttonpress(0) == True:
            plt.close()
            break