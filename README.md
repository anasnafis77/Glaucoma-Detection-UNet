# Glaucoma Detection Using U-Net and Multiple Localization

## Intoduction:

Glaucoma is the second cause of blindness. Unfortunately, Indonesian people still unaware for the importance of early glaucoma detection. Moreover, Glaucoma Detection is the laborous and subjective job for ophtalmologists. Thus, we need the automation for glaucoma detection. We use glaucoma detection based on Optic Cup and Disc segmentation. This automation allow us to detect glaucoma more faster and less subjective. Furthermore, this system could be implemented in smartphone, hence more patient could monitor their glaucoma stage in much more affordable way. For further algorithm explanation, you can read this paper code [here](http://www.joig.net/index.php?m=content&c=index&a=show&catid=78&id=299) 

Below are the main algorithm for Glaucoma Detection used in this project: 
1. Preprocessing
2. Optic Disc Localization
3. Optic Disc dan Cup Segmentation
4. Glaucoma Feature Extraction
5. Classification

## Usage
Below are the procedure for using this script:
1. Run this: `python inference_script.py` in command prompt. Make sure you are in 'Code' directory.
2. Choose your retinal image.
3. After a few seconds, the detection result should appear in command prompt and json file.
 
## Notebooks:
Main Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Notebooks/Main_notebook.ipynb) 

Localization Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Notebooks/Localization_Notebook.ipynb)

Segmentation Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Notebooks/Segmentation_notebook.ipynb)

Inferencing Notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anasnafis77/Deteksi-Glaukoma/blob/main/Notebooks/Inferencing_notebook.ipynb)

## Reference
A. N. Almustofa, A. Handayani, and T. L. R. Mengko, "Optic Disc and Optic Cup Segmentation on Retinal Image Based on Multimap Localization and U-Net Convolutional Neural Network," Journal of Image and Graphics, Vol. 10, No. 3, pp. 109-115, September 2022.
[link](http://www.joig.net/index.php?m=content&c=index&a=show&catid=78&id=299) 
