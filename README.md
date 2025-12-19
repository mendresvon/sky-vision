---
title: Sky Vision Classifier
emoji: 🦸🏻‍♂️
colorFrom: blue
colorTo: red
sdk: gradio
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# 🦸🏻‍♂️ Sky Vision Classifier | 空中物體辨識系統

### 🚀 Live Demo

Try the model here:
[Hugging Face Space Link](https://huggingface.co/spaces/breznev/bird-plane-superman)

### 👤 Developer Information

- **Name / 姓名:** 馬盛中 (Ma Sheng-Zhong)
- **Student ID / 學號:** 4B1YZ001
- **Institution / 學校:** Southern Taiwan University of Science and Technology (STUST)
- **Department / 系所:** Computer Science and Information Engineering (CSIE)

---

## 📖 Project Overview

This project is a deep learning-based image classifier inspired by the classic phrase _"It's a
bird... It's a plane... It's Superman!"_. The model was trained using the **fastai** framework to
distinguish between these three specific aerial categories with high confidence.

### 🎯 Supported Classes

The model is optimized to identify the following classes:

1. **Bird** (鳥)
2. **Plane** (飛機)
3. **Superman** (超人)

---

## 🛠️ Technical Stack

- **Architecture:** ResNet34 (Transfer Learning)
- **Framework:** fastai v2 / PyTorch
- **Deployment:** Gradio & Hugging Face Spaces
- **Language:** Python 3.11
- **Dataset:** Custom dataset scraped from Google Images using `icrawler`

---

## 🚀 How to Use

1. **Upload:** Drag and drop an image of a bird, plane, or Superman into the input box.
2. **Analyze:** Click the "Classify Image" button.
3. **Results:** View the top 3 most likely categories and their corresponding confidence scores.

---

## 🎓 Academic Context

This project was developed as part of a deep learning coursework at **STUST CSIE**. It demonstrates
the complete machine learning pipeline:

- **Data Collection:** Automated scraping of images using `icrawler` with specific search queries
  (e.g., "eagle flying", "commercial airplane", "henry cavill superman").
- **Data Cleaning:** Automated verification and removal of corrupt images using fastai's
  `verify_images`.
- **Model Training:** Fine-tuning a pre-trained ResNet34 model.
- **Deployment:** Full-stack deployment using Gradio.
