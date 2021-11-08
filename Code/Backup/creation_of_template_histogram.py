# importing the module
import cv2
import os
import pandas as pd
import csv
import numpy as np
from matplotlib import pyplot as plt

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    global centroid
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        temp = img
        cv2.putText(temp, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', temp)

        # checking for right mouse clicks

    centroid = [x, y]


# Fungsi untuk mengekstrak ROI
# input : list centroid, panjang sisi ROI (s), dan image source
# output : ROI image
def ekstrakROI(centroid, s, src):
    center = centroid

    h, w, _ = src.shape

    x = int(center[0])
    y = int(center[1])
    x0 = round(x - 0.5 * s)
    x1 = round(x + 0.5 * s)
    y0 = round(y - 0.5 * s)
    y1 = round(y + 0.5 * s)

    # penanganan kasus out of image
    if (x0 < 0):
        x1 = x1 + (-x0)
        x0 = x0 + (-x0)
    elif (x1 > w - 1):
        x0 = x0 - (x1 - (w - 1))
        x1 = x1 - (x1 - (w - 1))

    if (y0 < 0):
        y1 = y1 + (-y0)
        y0 = y0 + (-y0)
    elif (y1 > h - 1):
        y0 = y0 - (y1 - (h - 1))
        y1 = y1 - (y1 - (h - 1))
    # cropping ROI from source image
    ROI = src[y0:y1, x0:x1]

    return ROI

if __name__ == "__main__":
   # mengecilkan ukuran
    mypath = 'Images/Training'
   # dataframe train image (for template histogram purpose)
    full_path_tr = []
    file_name_tr = []
    centroid_tr = []

   # dataframe test image (only for test purpose)
   # full_path_ts = []
   # file_name_ts = []
   # centroid_ts = []
   # inisisasi template histogram
    temp = cv2.imread('image.png')
    mask = np.zeros(temp.shape[:2], np.uint8)
    sumHist = cv2.calcHist(mask, [1], None, [256], [0, 256])
    templateHist = cv2.calcHist(mask, [1], None, [256], [0, 256])

    # memasukkan path dan nama ke dataframe training dan test

    for path,subdir,files in os.walk(mypath):
        for name in files:
            full_path_tr.append(os.path.join(path, name))
            file_name_tr.append(name)
        break

    # masukan variabel yang sudah dikumpulkan pada looping di atas menjadi sebuah dataframe agar rapih
    df_tr = pd.DataFrame({"path": full_path_tr, 'file_name': file_name_tr})

    # Cek apakah file centroid image training.csv sudah diisi
    isFilled = os.path.getsize("centroid image training.csv")
    if isFilled == 0:
        # input centroid of image
        for i in range(len(df_tr["path"])):
             cv2.namedWindow('image', cv2.WINDOW_NORMAL)
             img = cv2.imread(df_tr["path"][i])
             # displaying the image
             cv2.imshow('image', img)

             # setting mouse hadler for the image
             # and calling the click_event() function

             centroid = []
             cv2.setMouseCallback('image', click_event)

             # wait for a key to be pressed to exit
             cv2.waitKey(0)

             # close the window
             cv2.destroyAllWindows()
             centroid_tr.append(centroid)

         # save nilai centroid ke csv
        df_tr['center'] = centroid_tr
        df_tr.to_csv('centroid image training.csv', index=False, header=False)
    else:
        # open centroid data
        with open('centroid image training.csv', newline='') as f:
             reader = csv.reader(f)
             data = pd.DataFrame(list(reader))
             centroid_tr = data[3][1:].tolist()
             for i in range(len(centroid_tr)):
                 centroid_tr[i] = centroid_tr[i].replace("[", "")
                 centroid_tr[i] = centroid_tr[i].replace("]", "")
                 x, y = centroid_tr[i].split(",")
                 centroid_tr[i] = [int(x), int(y)]



    # membuat histogram template dari ROI image training
    for i in range(len(centroid_tr)):
         # Ekstrak ROI
         s = 500
         img = cv2.imread(df_tr["path"][i])
         ROI = ekstrakROI(centroid_tr[i], s, img)
         # Summing Histogram
         hist = cv2.calcHist(ROI, [1], None, [256], [0, 256]) # histogram of green channel
         sumHist = sumHist + hist

    ndata = len(df_tr["path"]) #banyaknya data gambar, sebagai pembagi nilai bin histogram

    templateHist = sumHist/ndata
    templateHistogram = np.round(templateHist)
     # save templateHist to CSV
     # opening the csv file in 'w+' mode
    files = open('template histogram.csv', 'w+', newline = '')
    with files:
         writer = csv.writer(files)
         writer.writerows(templateHistogram)

    plt.plot(templateHistogram)
    plt.show()


