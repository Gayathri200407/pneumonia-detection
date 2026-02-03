# Pneumonia Detection Using Deep Learning

A deep learning–based image classification project to detect **pneumonia from chest X-ray images** using **FastAI** and **PyTorch**. The system classifies X-rays as *Normal* or *Pneumonia* by leveraging transfer learning with a convolutional neural network (CNN).

---

## Objective
To build an end-to-end deep learning pipeline for medical image classification, focusing on data preprocessing, model training, evaluation, and inference.

---

## Tech Stack
- **Python**
- **FastAI**
- **PyTorch**
- **Torchvision**
- **NumPy / Matplotlib**

---

## Dataset
- Chest X-ray image dataset
- Two classes: **Normal** and **Pneumonia**
- Images organized using a folder-based structure compatible with FastAI’s data loaders

> Dataset is not included in the repository due to size constraints.

---

## Approach
- Loaded and preprocessed chest X-ray images using FastAI’s `ImageDataLoaders`
- Applied image resizing and normalization
- Used **transfer learning** with a pretrained **ResNet18** CNN
- Fine-tuned the model on the training dataset
- Evaluated performance using accuracy metrics
- Implemented inference to predict pneumonia on unseen X-ray images

---

## Features
- Automated image preprocessing
- CNN-based classification using transfer learning
- Model evaluation with accuracy metrics
- Prediction on new/unseen chest X-ray images

---


