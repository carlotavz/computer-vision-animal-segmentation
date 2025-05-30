# Wildlife Image Classification with Segmentation-Aided CNNs

This project focuses on **classifying wildlife images** (elephant, rhino, flamingo, emu) using semantic segmentation to enhance the performance of a convolutional classifier. It combines classical machine learning, U-Net segmentation, and CNN-based classification into a clean, modular pipeline.

---

## Objective

Predict **what animal appears in an image**, using tools like U-Net to guide the classifier by focusing on the relevant region of interest.

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
│   └── 03_model_training.ipynb
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
├── main.py
├── requirements.txt          # Dependencies list
└── .gitignore                # Files and folders to ignore in git
```


---

## Pipeline Overview

1. **EDA & Mask Quality Checks**  
2. **Semantic Segmentation**  
   - U-Net in PyTorch  
   - Random Forest / XGBoost baseline
3. **Mask-guided Preprocessing**  
   - Full image
   - Mask as 4th channel
   - Cropped ROI (Region of Interest)
4. **CNN Image Classification**  
5. **Evaluation & Visualization**  

---

## Models Included

| Model                | Purpose              | Notes                              |
|---------------------|----------------------|-------------------------------------|
| `UNetWrapper`        | Segmentation         | PyTorch implementation of U-Net     |
| `ClassicModel`       | Segmentation (ML)    | Random Forest / XGBoost             |
| `SimpleCNNClassifier`| Classification       | 3-layer CNN                         |
| `ConvNetBaseline`    | Segment. baseline    | Simple encoder-decoder conv net     |

---

## Metrics Tracked

- **Accuracy**
- **F1 Score (macro/micro)**
- **Precision / Recall**
- **IoU, Dice (for segmentation)**
- **Confusion Matrix**

---

## Sample Results

| Method                      | Accuracy | F1 Score | Precision | Recall |
|-----------------------------|----------|----------|-----------|--------|
| CNN (raw image)             | 82.4%    | 0.81     | 0.83      | 0.80   |
| CNN + U-Net (ROI focused)   | 89.2%    | 0.88     | 0.89      | 0.87   |
| CNN + Classic Mask          | 85.0%    | 0.84     | 0.85      | 0.83   |

---

## Visual Example

| Input Image | U-Net Mask | ROI Cropped | Predicted Class |
|-------------|------------|-------------|-----------------|
| ✅          | ✅         | ✅          | ✅               |

---


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

## Models

Model	Type	Notes
Random Forest	Classical ML	Trained on flattened pixel vectors
XGBoost	Classical ML	Trained per-pixel using patch statistics
U-Net	Deep Learning	Full image segmentation (PyTorch)


## Acknowledgements

U-Net architecture: Ronneberger et al. (2015)
OpenCV, PyTorch, Scikit-learn
Inspiration from segmentation benchmarks and ecological AI applications
## Author
[Carlota Vázquez Arrojo] – Data Scientist | Computer Vision Specialist
Contact: [www.linkedin.com/in/carlota-vazquez-arrojo-b3a639213]

## License

MIT License – feel free to fork, adapt, and extend.
