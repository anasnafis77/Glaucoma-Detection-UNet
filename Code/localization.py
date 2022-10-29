import cv2
import math
import numpy as np
from skimage.transform import resize
from skimage.segmentation import slic
class OD_localization():
  def __init__(self):
    # loading template
    image_R = cv2.imread('Templates/ROItemplateRed.png', 0)
    image_G = cv2.imread('Templates/ROItemplateGreen.png', 0)
    image_B = cv2.imread('Templates/ROItemplateBlue.png', 0)
    self.image_templates = [image_R, image_G, image_B]

  def preprocessing(self, src, std_size):
    return cv2.resize(src, std_size, interpolation=cv2.INTER_CUBIC)

  def extract_BR_map(self, src, mask, numSegments=50, sigma=10):
    """
      This is the function of brightness map of retinal images extraction. 
      Brightness map is an image that contain well-segmented areas of retinal images
      that filled with the brightness of those areas. This maps give the brightness
      information of all area in retinal image. Since the Optic Disc typically
      the one of brightest area in retinal images, this map helps identify the
      location of Optic Disc.
      
      Input:
      src: retinal images (RGB)
      mask: mask for Masking out non-retinal area of the image
      numSegments: The number of segment in the brightness map
      sigma: the size of gaussian filter for blurring the image

      Output:
      Brightnes map: the combined brightness across R, G, B channels
    """

    #Resize imagen source
    H, W = src.shape[:2]
    
    # Resizing the image to quarter size for accelerating the computation
    resized_src = resize(src, output_shape=(H//4, W//4), mode = 'constant')
    resized_mask =  resize(mask, output_shape=(H//4, W//4), mode = 'constant')

    # Blurring the image tim improve the segment accuracy
    blur = cv2.GaussianBlur(resized_src, (37, 37), 0)

    # Segmenting the retinal image using SLIC algorithm 
    segments = slic(blur, n_segments=numSegments, sigma=sigma, mask=resized_mask)
    # boundary = mark_boundaries(resized_src, segments)

    labels = np.unique(segments)
    h, w = resized_src.shape[:2]

    # initiate the brightness map of R, G, and B channel
    g_map = np.zeros([h, w], np.float32)
    r_map = np.zeros([h, w], np.float32)
    b_map = np.zeros([h, w], np.float32)


    r_src = resized_src[:, :, 0]
    g_src = resized_src[:, :, 1]
    b_src = resized_src[:, :, 2]

    # Filling the segment with brightness of it
    for label in labels:
      # Brightness is define as the maximum intensity of each segment
      r_map[segments==label] = np.max(r_src[segments==label])
      g_map[segments==label] = np.max(g_src[segments==label])
      b_map[segments==label] = np.max(b_src[segments==label])

    # Resize back the brightness map to its original size
    r_resized_map = resize(r_map, output_shape=(H, W), mode='constant', preserve_range=True)
    g_resized_map = resize(g_map, output_shape=(H, W), mode='constant', preserve_range=True)
    b_resized_map = resize(b_map, output_shape=(H, W), mode='constant', preserve_range=True)

    # combining brightness map using average method
    combined_brightness_map = (r_resized_map + g_resized_map + b_resized_map)/3
    
    return combined_brightness_map

  def padding(self, NCC, h, w):
    """
      This function will zero padding the NCC map such that NCC map have the same size
      with retinal images
    """

    ver = (h - NCC.shape[0])/2
    hor = (w - NCC.shape[1])/2
    top = math.floor(ver)
    bottom = math.ceil(ver)
    left =  math.floor(hor)
    right = math.ceil(hor)
    value = 0
    borderType = cv2.BORDER_CONSTANT
    zeropadded_NCC = cv2.copyMakeBorder(NCC, top, bottom, left, right, borderType, value)
    return zeropadded_NCC

  def image_template_matching(self, clahe_retinal_img, template, mask):
    """
      This is a function for OD template matching. This will yields Normalized
      Correlation Coefficient (NCC) of R, G, and B channel. 

      Input:
      clahe_retinal_img: RGB retinal image that had been CLAHE-ed
      template: OD template image 
      mask: Mask for masking out the non-retinal area

      Output: 
      NCC_maps: NCC image of R, G, and B channel
    """

    h, w = clahe_retinal_img[0].shape[:2]
    NCC_maps = []
    for i, clahe_image in enumerate(clahe_retinal_img):
      NCC = cv2.matchTemplate(clahe_image, template[i], cv2.TM_CCOEFF_NORMED)
      NCC = NCC + abs(np.min(NCC))
      NCC = self.padding(NCC, h, w)
      NCC = NCC * mask

      NCC_maps.append(NCC)
    
    return NCC_maps


  def locate(self, src, coeff_args, test_on=False):
    """
      This is the main function for localization of Optic Disc
      
      Input:
      src: retinal images (RGB)
      template_images: Optic Disc template images of R, G, B channles
      used_channels: defining the used NCC channels for localization
      NCC_on : Use NCC map for localization
      bright_on : Use brightness map for localization
      r_coeff: coefficient of red_NCC map 
      g_coeff: coefficient of green_NCC map 
      b_coeff: coefficient of blue_NCC map 
      br_coeff: coefficient of brightness map 

      Output:
      disk_center: Optic Disc Center
      all_maps: list of created maps for localization, returned only if 
      test_on is True.
    """

    R_COEFF, G_COEFF, B_COEFF, BR_COEFF = coeff_args
    h, w = src.shape[:2]

    # Implement CLAHE to the input image 
    clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
    cl_img = clahe.apply(src[:, :, 1])
    
    zeros = np.zeros([h, w], np.uint8)
    img = [zeros, zeros, zeros]
    
    # Apply CLAHE to each of image channel 
    img[0] = clahe.apply(src[:, :, 0])
    img[1] = clahe.apply(src[:, :, 1])
    img[2] = clahe.apply(src[:, :, 2])
      
    # Mask Out outside retinal image
    blurred = cv2.GaussianBlur(cl_img, (7, 7), 0)
    (T, mask) = cv2.threshold(blurred, 10 , 255, cv2.THRESH_BINARY)
    mask = mask/255

    # Extract the NCC maps
    red_NCC, green_NCC, blue_NCC = self.image_template_matching(img, self.image_templates, mask)

    # Extract superpixel map
    brightness_map = self.extract_BR_map(src, mask)

    # Combining localization maps 
    combined_map = red_NCC * R_COEFF + green_NCC*G_COEFF + blue_NCC*B_COEFF + brightness_map*BR_COEFF
    all_maps = [combined_map, red_NCC, green_NCC, blue_NCC, brightness_map]
    
    # Extracting maximum value of NCC
    y, x = list(zip(*np.where(combined_map==np.max(combined_map))))[0]
    disk_center = (x, y)

    # Return all maps for testing the algorithm only
    if test_on:
      return disk_center, all_maps 
    elif not test_on:
      return disk_center
