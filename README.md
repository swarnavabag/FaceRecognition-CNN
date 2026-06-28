# FaceRecognition-CNN

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)
![CNN](https://img.shields.io/badge/CNN-Custom%20ResNet-green?style=for-the-badge)
![ArcFace](https://img.shields.io/badge/Loss-ArcFace-orange?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-yellow?style=for-the-badge&logo=opencv)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A modular PyTorch-based face recognition system with preprocessing, CNN training, evaluation, and inference pipeline for deep learning research and experimentation.

---

## Project Overview

This project implements a deep learning-based face recognition pipeline using a **custom ResNet-style CNN** trained with **ArcFace loss** for robust face verification and recognition.

The model learns discriminative embeddings such that:
- Faces of the **same identity** are closer in embedding space
- Faces of **different identities** are farther apart

This enables identity verification using similarity-based matching.

---

## Features

- Modular project structure
- Custom ResNet-style CNN backbone
- ArcFace metric learning
- 512-dimensional embeddings
- Image preprocessing pipeline
- Training & evaluation workflow
- Similarity-based face verification
- Notebook-based experimentation

---

## Model Architecture

### Backbone
Custom **ResNet-inspired CNN** with residual connections for hierarchical feature extraction.

### Embedding Layer
- Embedding size: **512**

### Metric Learning
**ArcFace Loss** improves:
- Inter-class separation
- Intra-class compactness
- Verification robustness

---

## Dataset

Training was performed on a subset of **VGGFace2**.

### Dataset Split
- Training identities: **~480**
- Validation identities: **~60**

Dataset includes variations in:
- Pose
- Illumination
- Expression
- Age
- Occlusion

> Dataset is not included in this repository due to size and licensing constraints.

---

## Results

### Training Performance
- Train Accuracy: **95.05%**

### Verification Performance
- Verification Accuracy: **84.92%**
- Best Threshold: **0.15**

### Embedding Similarities

#### Train Set
- Positive Similarity: **0.7908**
- Negative Similarity: **-0.0041**

#### Validation Set
- Positive Similarity: **0.3436**
- Negative Similarity: **0.00385**

### Observations
- Strong clustering of training identities
- Good separation of negative pairs
- Validation gap suggests room for better generalization

---

## Project Structure

```bash
FaceRecognition-CNN/
│
├── models/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_architecture&training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── losses.py
│   ├── model.py
│   ├── preprocessing.py
│   └── train.py
│
├── requirements.txt
├── README.md
└── .gitignore
````

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/swarnavabag/FaceRecognition-CNN.git
cd FaceRecognition-CNN
````

---

### Create Virtual Environment (Python 3.11 Recommended)

```bash
python -m venv .venv
```

Activate the environment:

#### Windows (PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

#### Windows (CMD)

```cmd
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

---

### Install PyTorch

Install PyTorch separately based on your system configuration (CPU or CUDA-enabled GPU).

Refer to the official installation guide:

https://pytorch.org/get-started/locally/

Example (CUDA 12.1):

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Example (CPU only):

```bash
pip3 install torch torchvision torchaudio
```

---

### Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

### Verify Installation

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Training

```bash
python src/train.py
```

---

## Evaluation

```bash
python src/evaluate.py
```

---

## Inference

Inference pipeline:

1. Load face image
2. Generate embedding
3. Compare against known embeddings
4. Apply similarity threshold

---

## Future Improvements

* Integrate face detector (YOLO / MTCNN / RetinaFace)
* Add face alignment
* Improve generalization
* Compare multiple CNN backbones
* Real-time webcam inference
* Web deployment

---

## Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy
* Jupyter Notebook

---

## License

Licensed under the MIT License.

