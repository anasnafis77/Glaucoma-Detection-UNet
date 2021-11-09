import cv2
import numpy as np
from skimage.transform import resize

# mengembalikan size Mask OC dan OD menjadi size citra awal
def resizeMask(mask, koordinat, shape):
  yo, yi, xo, xi = koordinat

  mask_temp = np.zeros(shape, np.uint8)

  for y in range(yo, yi):
    for x in range(xo, xi):
      if mask[y-yo][x-xo] == 255:
        mask_temp[y][x] = 255

  return mask_temp


# Circle Hough Transform Model fitting
def ellipsTransform(mask):
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

def semantic_segmentation(ROI, model, coordinate, shape, elips_fit=True, ):
  ROI = resize(ROI, (256, 256, 1), mode = 'constant', preserve_range = True)
  ROI = np.array([ROI/255.0])
  pred = model.predict(ROI)
  seg_bin = np.array(pred > 0.5, np.uint8)
  seg_bin = seg_bin[0].squeeze()
  seg_bin = seg_bin*255

  # resizing the mask to its original size
  seg_bin = resizeMask(seg_bin, coordinate, shape)

  # Apply elipse fitting if elips_fit is True
  if elips_fit:
    seg_bin = ellipsTransform(seg_bin)

  return seg_bin