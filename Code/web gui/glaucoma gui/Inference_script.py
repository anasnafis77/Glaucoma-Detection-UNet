import cv2 
import sys
import time
import os
import math
import pickle
import numpy as np
import tkinter as tk
import tensorflow as tf
from skimage.transform import resize
from skimage.segmentation import slic
from matplotlib import pyplot as plt
from keras import backend as K
from localization import *
from segmentation import *
from classification import *
from supporting import *

def inference(file_path):
    # Load model
    segmentation_model_path_OD = 'Code/Models/model OD semantic'
    segmentation_model_path_OC = 'Code/Models/model OC semantic'
    inference_model_path = 'Code/Models/Inference model/inference_model.sav'

    # Load semantic segmentation model 
    model_OD = tf.keras.models.load_model(segmentation_model_path_OD,custom_objects={'fscore':fscore})
    model_OC = tf.keras.models.load_model(segmentation_model_path_OC,custom_objects={'fscore':fscore})

    # Load inference model
    inference_model = pickle.load(open(inference_model_path, 'rb'))

    # Load Optic Disc Image Templates
    image_R = cv2.imread('Code/Templates/ROItemplateRed.png', 0)
    image_G = cv2.imread('Code/Templates/ROItemplateGreen.png', 0)
    image_B = cv2.imread('Code/Templates/ROItemplateBlue.png', 0)
    image_templates = [image_R, image_G, image_B]

    # Define Hyperparameter 
    ROI_size = 550
    r_coeff, g_coeff,b_coeff, br_coeff = 1, 0.2, 0, 0.8
    used_channels = 'rg' 

    # load image and ROI 
    retinal_image = cv2.imread(file_path, 1)
    retinal_image = cv2.cvtColor(retinal_image, cv2.COLOR_BGR2RGB)

    # Localize Optic Disc
    start_loc = time.time()
    disc_center = OD_Localization(retinal_image, image_templates, 
                                used_channels=used_channels, 
                                bright_on=True,  
                                r_coeff=r_coeff, 
                                g_coeff=g_coeff, 
                                b_coeff=b_coeff,
                                br_coeff=br_coeff)
    end_loc = time.time()

    # OD and OC segmentation 
    start_seg = time.time()
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(retinal_image[:, :, 1])
    ROI, coordinate = ekstrakROI(disc_center, ROI_size, cl_img)
    OD_mask = semantic_segmentation(ROI, model_OD, coordinate, retinal_image.shape[:2])
    OC_mask = semantic_segmentation(ROI, model_OC, coordinate, retinal_image.shape[:2])
    end_seg = time.time()

    print('Localization time: {:.2f} s'.format(end_loc-start_loc))
    print('Segmentation time: {:.2f} s'.format(end_seg-start_seg))

    # Inference the feature
    VCDR,_,ACDR = CDR_calc(OD_mask, OC_mask)
    feature = np.array([VCDR, ACDR])
    pred = glaucoma_predict(inference_model, feature)

    return pred
 
