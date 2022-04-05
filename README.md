# Glaucoma Detection Using U-Net and Multiple Localization

## Intoduction:

Glaucoma is the second cause of blindness. Unfortunately, Indonesian people still unaware for the importance of early glaucoma detection. Moreover, Glaucoma Detection is the laborous and subjective job for ophtalmologists. Thus, we need the automation for glaucoma detection based on Optic Cup and Disc segmentation. This automoation allow us to detect glaucoma more faster. Furthermore, this is possible to implemented in smartphone, hence more people could monitor their eyes from glaucoma in much more affordable way.

Below are the main algorithm for Glaucoma Detection used in this project: 
1. Preprocessing
2. Optic Disc Localization
3. Optic Disc dan Cup Segmentation
4. Glaucoma Feature Extraction
5. Classification

## Usage
Below are the procedure for using this script:
1. Run this: `python inference_script.py` in command prompt. Make sure you are on 'Code' the directory.
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
