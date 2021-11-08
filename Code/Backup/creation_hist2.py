import cv2
from matplotlib import pyplot as plt
import os
import pandas as pd
import csv


img1 = cv2.imread('image.png')
img2 = cv2.imread('image2.png')
h1 = cv2.calcHist(img1, [1], None, [256], [0, 256])
h2 = cv2.calcHist(img2, [1], None, [256], [0, 256])
h3 = h1 + h2

print(h1.shape)
print(h2.shape)
print(h3.shape)

plt.subplot(311), plt.plot(h1)
plt.subplot(312), plt.plot(h2)
plt.subplot(313), plt.plot(h3)

plt.show()