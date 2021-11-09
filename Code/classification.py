import cv2
import numpy as np 
def CDR_calc(OD_mask, OC_mask):
  try:
    c_OD,_ = cv2.findContours(OD_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    p_OD = cv2.approxPolyDP(c_OD[0], 3, True)
    c_OC,_ = cv2.findContours(OC_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    p_OC = cv2.approxPolyDP(c_OC[0], 3, True)
    x_OD, y_OD, hor_OD, ver_OD = cv2.boundingRect(p_OD)
    x_OC, y_OC, hor_OC, ver_OC = cv2.boundingRect(p_OC)
    area_OD = np.sum(OD_mask == 255)/255
    area_OC = np.sum(OC_mask == 255)/255

    VCDR = ver_OC/ver_OD # vertical CDR
    HCDR = hor_OC/ hor_OD # Horizontal CDR
    ACDR = area_OC/ area_OD # Area CDR

  except:
    VCDR = float("nan")
    HCDR = float("nan")
    ACDR = float("nan")
    
  return VCDR, HCDR, ACDR


def glaucoma_predict(model, feature):
  predict = model.predict([feature])
  if predict[0] == 1:
    return 'Glaucoma'
  elif predict[0] == 0:
    return 'Normal'
