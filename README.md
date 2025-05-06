# Deep Learning for Pediatric Pneumonia Detection Using Chest X-Ray Images

**Author:** Mwidy Mounange-Badimi

**Course:** BMI 6334 – Deep Learning in Biomedical Informatics

## Overview

A transfer-learning pipeline that classifies pediatric chest X-rays as **NORMAL** or **PNEUMONIA**:

- Lung-region cropping with Otsu thresholding + contour masking (OpenCV)
- Data augmentation with Albumentations (CLAHE, brightness/contrast, shift/scale/rotate, flips, noise, blur)
- Stratified 80/10/10 train/validation/test split
- Xception backbone (ImageNet weights) with a dense head, trained with binary focal loss and mixed precision
- Two-phase training: frozen-backbone warm-up, then full fine-tuning at a lower learning rate
- Diagnostics: ROC curve, confusion matrix, F1 vs. threshold, calibration curve, false-positive/negative gallery

## Results (held-out test set, 586 images)

| Metric | Value |
|---|---|
| Accuracy | 0.962 |
| ROC-AUC | 0.992 |
| Recall (sensitivity) | 0.957 |
| Best validation AUC | 0.990 |

## Notebook

[`6334_Final_project_report_paper_Code.ipynb`](6334_Final_project_report_paper_Code.ipynb) – open it on GitHub to read the code and outputs, or use the **Open in Colab** badge at the top of the notebook to run it.

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) on Kaggle. Running the notebook requires a Kaggle API key (`kaggle.json`) and a GPU runtime (developed on a Colab A100).
