import cv2
import numpy as np
from skimage.transform import resize
from keras import backend as K
import tensorflow as tf

def fscore(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class optic_segmentation():
  def __init__(self, model_OD_path='Models/model OD semantic', model_OC_path='Models/model OC semantic'):
    # Load semantic segmentation model 
    self.model_OD = tf.keras.models.load_model(model_OD_path,custom_objects={'fscore':fscore})
    self.model_OC = tf.keras.models.load_model(model_OC_path,custom_objects={'fscore':fscore})

  # mengembalikan size Mask OC dan OD menjadi size citra awal
  def resizeMask(self, mask, koordinat, shape):
    yo, yi, xo, xi = koordinat

    mask_temp = np.zeros(shape, np.uint8)

    for y in range(yo, yi):
      for x in range(xo, xi):
        if mask[y-yo][x-xo] == 255:
          mask_temp[y][x] = 255

    return mask_temp


  # Circle Hough Transform Model fitting
  def ellipsTransform(self, mask):
    # Select largest contour
    cnts, _= cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    elips = np.zeros((mask.shape[0], mask.shape[1]),np.uint8)
    if len(cnts)>0:
    # Ellipse fitting
      ellipse = cv2.fitEllipse(cnts[0])
      elips = cv2.ellipse(elips, ellipse, 255, cv2.FILLED)
    else:
      elips = mask
    return elips

  def do_segmentation(self, ROI, coordinate, shape, elips_fit=True):
    ROI = resize(ROI, (256, 256, 1), mode = 'constant', preserve_range = True)
    ROI = np.array([ROI/255.0])
    # OD segmentation
    OD_pred = self.model_OD.predict(ROI)
    OD_pred = np.array(OD_pred > 0.5, np.uint8)
    OD_pred = OD_pred[0].squeeze()
    OD_pred = OD_pred*255
    # OC segmentation
    OC_pred = self.model_OC.predict(ROI)
    OC_pred = np.array(OC_pred > 0.5, np.uint8)
    OC_pred = OC_pred[0].squeeze()
    OC_pred = OC_pred*255
    
    # Apply elipse fitting if elips_fit is True
    if elips_fit:
      OD_pred = self.ellipsTransform(OD_pred)
      OC_pred = self.ellipsTransform(OC_pred)

    # resizing the mask to its original size
    OD_pred = self.resizeMask(OD_pred, coordinate, shape)
    OC_pred = self.resizeMask(OC_pred, coordinate, shape)


    return OD_pred, OC_pred