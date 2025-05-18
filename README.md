# Computer Vision for Animal Segmentation
**Pixel-wise segmentation of wildlife images using classical machine learning and U-Net**

## Overview

This project addresses the challenge of segmenting animals (elephant, rhino, emu, flamingo) from real-world images using both classical machine learning models and deep learning (U-Net). The pipeline has been designed to reflect production-level standards, focusing on modularity, reproducibility, and clarity. It includes data preprocessing, patch extraction, model training, evaluation, and visualization of results.

This repository serves both as a research experiment and as a portfolio project in the field of computer vision, with potential applications in ecology, conservation, and smart image annotation systems.

---

## Project Structure


```bash
computer-vision-animal-segmentation/
├── data/                     # Raw data: images and corresponding binary masks
│   ├── images/
│   │   ├── elephant/
│   │   ├── rhino/
│   │   ├── flamingo/
│   │   └── emu/
│   └── masks/
│       ├── elephant/
│       ├── rhino/
│       ├── flamingo/
│       └── emu/
│
├── notebooks/                # Exploratory notebooks and experiments
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_classic_ml_training.ipynb
│   └── 04_unet_training.ipynb
│
├── src/                      # Modular source code
│   ├── preprocessing/
│   │   ├── patch_extractor.py
│   │   ├── transformations.py
│   │   └── utils.py
│   ├── models/
│   │   ├── unet.py
│   │   └── classic_models.py
│   └── training/
│       ├── trainer.py
│       └── metrics.py
│
├── results/                  # Output visualizations and evaluation metrics
│   ├── visualizations/
│   ├── predictions/
│   └── metrics/
│
├── README.md                 # Project description and instructions
├── requirements.txt          # Dependencies list
└── .gitignore                # Files and folders to ignore in git
```

## Key Features

-  Clean and modular pipeline using Python
-  EDA on image/mask statistics (distribution, resolution, class balance)
-  Custom patch extractor for training efficiency
-  Training of classical models (Random Forest, XGBoost)
-  Deep learning pipeline with PyTorch U-Net
-  Pixel-wise evaluation using IoU, Dice, Precision, Recall
-  Side-by-side visualization of predictions vs. ground truth
-  Designed for scalability and adaptation to other datasets


## Dataset Description

- Images and binary masks grouped by animal species.
- Masks are perfectly aligned by filename with their corresponding images.
- Each image-mask pair is suitable for supervised semantic segmentation.

Example:
data/
├── images/elephant/image_0001.jpg
└── masks/elephant/mask_0001.png


---

## Setup Instructions

### 1. Clone the repository
git clone https://github.com/your_username/computer-vision-animal-segmentation.git
cd computer-vision-animal-segmentation
## 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
## 3. Install dependencies
pip install -r requirements.txt
You may need to install PyTorch manually depending on your hardware (CUDA or CPU).

## Metrics Used
Intersection over Union (IoU)
Dice Coefficient
Pixel Accuracy
Precision / Recall
Confusion Matrix (for binary masks)

## Models

Model	Type	Notes
Random Forest	Classical ML	Trained on flattened pixel vectors
XGBoost	Classical ML	Trained per-pixel using patch statistics
U-Net	Deep Learning	Full image segmentation (PyTorch)

## Notebooks

Notebook	Description
01_exploratory_data_analysis.ipynb	Initial EDA of image and mask characteristics
02_preprocessing_pipeline.ipynb	Preprocessing, patch extraction, and stats
03_classic_ml_training.ipynb	Training and evaluation of classical models
04_unet_training.ipynb	Full PyTorch U-Net training pipeline

## TODOs

 Add cross-validation with stratified patches
 Include Mask R-CNN as a comparative deep model
 Optimize model checkpointing and early stopping
 Experiment with data augmentation (albumentations)

## Acknowledgements
Data collected and manually verified for accuracy
Inspiration from segmentation challenges and academic examples
Deep learning architecture inspired by the original U-Net paper (Ronneberger et al., 2015)

## Author
[Carlota Vázquez Arrojo] – Data Scientist | Computer Vision Specialist
Contact: [www.linkedin.com/in/carlota-vazquez-arrojo-b3a639213]

## License

MIT License – feel free to fork, adapt, and extend.
