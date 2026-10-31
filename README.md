# ANEMEIA — U-Net and EfficientNet-B0 Based Deep Learning Framework for Non-Invasive Anemia Screening from Conjunctival Images

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 1. Project Overview

**ANEMEIA** is an end-to-end medical computer vision and deep learning framework designed for non-invasive, preliminary anemia screening. The system analyzes close-up photographs of the human lower palpebral conjunctiva (the vascular mucous membrane lining the inner eyelid) to assess microvascular hemoglobin concentration without requiring needles, finger-pricks, or laboratory blood draws.

The solution integrates automated semantic conjunctival segmentation via **U-Net**, region-of-interest (ROI) standardization, fine-tuned transfer learning classification via **EfficientNet-B0**, visual explainability via **Grad-CAM**, and an interactive healthcare web dashboard with nutritional guidance.

---

## 2. Problem Statement

Anemia is a major global public health concern affecting over 1.6 billion individuals (~25% of the global population), disproportionately impacting children, pregnant women, and communities in low- and middle-income regions.

- **Limitations of Conventional Diagnosis:** Standard diagnosis relies on venipuncture-based Complete Blood Count (CBC) or point-of-care invasive hemoglobinometers. These methods require sterile consumables, trained phlebotomists, laboratory infrastructure, cold-chain transport, and cause patient discomfort.
- **Subjectivity of Clinical Inspection:** While conjunctival pallor has long been recognized as a physical manifestation of low hemoglobin, manual inspection by healthcare providers exhibits high inter-observer variability and low diagnostic sensitivity for mild-to-moderate cases.
- **The Need for Automated Screening:** A non-invasive, automated, objective screening tool accessible via standard cameras provides an opportunity for point-of-care triage and community health screening.

---

## 3. Objectives

1. **Automated Palpebral Conjunctiva Segmentation:** Eliminate manual cropping by employing a custom U-Net to isolate the palpebral conjunctiva tissue from full eye images.
2. **Quality-Assured ROI Extraction:** Deterministically validate tissue presence, filter spurious artifacts, and normalize the Region of Interest (ROI) into standardized \(224 \times 224 \times 3\) inputs.
3. **Deep Transfer Learning Classification:** Leverage an EfficientNet-B0 backbone fine-tuned on clinical cohorts to classify conjunctival pallor as Anemic or Non-Anemic against an optimized validation decision threshold.
4. **Visual Interpretability:** Provide Gradient-Weighted Class Activation Mapping (Grad-CAM) heatmaps so healthcare providers can visually inspect which anatomical tissue regions influenced the model's decision.
5. **Holistic Healthcare Support:** Deliver a web-based clinical dashboard that couples AI screening with evidence-based dietary education and nutritional guidance.

---

## 4. Technologies Used

- **Deep Learning Framework:** TensorFlow 2.x, Keras, TensorFlow Lite (XNNPACK CPU acceleration delegate)
- **Computer Vision & Image Processing:** OpenCV (`cv2`), Pillow, NumPy, Matplotlib
- **Machine Learning & Evaluation:** Scikit-learn, Pandas, SciPy
- **Backend & Web Application:** Python 3.11, Flask, Werkzeug, Jinja2
- **Frontend Architecture:** HTML5, CSS3 (Modern Responsive Healthcare Design System), Vanilla JavaScript
- **Configuration & Reproducibility:** PyYAML, deterministic random seeds

---

## 5. System Architecture & Workflow

```
[ Uploaded Eye Photo (RGB) ]
             │
             ▼
[ In-Memory Image Preprocessing & RGB Standardization ]
             │
             ▼
[ 1. Palpebral Conjunctiva Segmentation (U-Net) ]
             │   ├── 256x256 Input Resolution
             │   └── Probability Mask Generation (Threshold = 0.5)
             ▼
[ 2. Deterministic Quality Check & ROI Extraction ]
             │   ├── Minimum Pixel & Connected-Component Filtering
             │   └── Adaptive Bounding Box Extraction + Margins (224x224)
             ▼
[ 3. Deep Classification (EfficientNet-B0) ]
             │   ├── Fine-Tuned Backbone + Dense Head
             │   └── Calibrated Decision Cutoff (Threshold = 0.3837)
             ▼
[ 4. Explainability & Triage Dashboard ]
             ├── Grad-CAM Saliency Heatmap
             ├── 4-Panel Visual Audit (Input, Mask, ROI, Heatmap)
             ├── Risk Percentage & Model Confidence
             └── Independent Anemia Awareness & Nutritional Guidance Section
```

---

## 6. Project Modules

- **`src/segmentation/`**: U-Net architecture definitions, binary cross-entropy loss, Dice coefficient metric, mask inference engine, and TensorFlow Lite runtime adapter.
- **`src/roi/`**: Automated bounding-box localization, margin padding, connected-component analysis, and quality validation.
- **`src/classification/`**: EfficientNet-B0 model construction, stage-wise transfer learning, cosine/polynomial learning rate scheduling, and threshold calibration.
- **`src/explainability/`**: Grad-CAM implementation calculating gradients of the target class score with respect to the last convolutional feature maps (`top_conv`).
- **`src/common/`**: Configuration management (`yaml`), clinical metrics calculation (Sensitivity, Specificity, F1, ROC-AUC), and visual utilities.
- **`templates/`**:
  - `index.html`: Responsive patient assessment interface, 4-panel visual result explainability, and separate nutritional guidance.
  - `dashboard.html`: Comprehensive model evaluation dashboard displaying confusion matrices, ROC curves, and split metrics.

---

## 7. Folder Structure

```
ANEMEIA/
├── app.py                     # Primary Flask web application and screening server
├── requirements.txt           # Verified Python package dependencies
├── run_app.bat                # Windows launcher script for automated server startup
├── README.md                  # Comprehensive project documentation
├── ARCHITECTURE_ML.md         # Machine learning pipeline specification
├── .gitignore                 # Exclusion rules for virtualenvs, caches, and datasets
│
├── configs/                   # Hyperparameter and pipeline YAML configurations
│   ├── classifier_efficientnetb0.yaml
│   ├── segmentation_unet.yaml
│   ├── data.yaml
│   └── labels.yaml
│
├── artifacts/                 # Serialized model checkpoints & evaluation reports
│   └── current/
│       ├── classifier/
│       │   ├── best_classifier.keras          # Fine-tuned EfficientNet-B0 model
│       │   ├── evaluation_dashboard_data.json # Verified benchmark evaluation metrics
│       │   └── threshold_config.json          # Optimal decision cutoff configuration
│       └── segmentation/
│           ├── best_unet.keras                # Trained Keras U-Net segmentation model
│           └── best_unet_opt.tflite           # Optimized TFLite XNNPACK inference engine
│
├── data/
│   └── splits/                # Patient-disjoint split definitions (Train, Val, Test)
│       ├── split_v1.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── scripts/                   # Training, inference, and evaluation automation scripts
│   ├── evaluate_end_to_end.py
│   ├── make_predicted_rois.py
│   ├── prepare_dataset.py
│   ├── train_classifier.py
│   └── train_segmentation.py
│
├── src/                       # Core Python library implementation
│   ├── classification/        # EfficientNet-B0 model, dataset loaders, training logic
│   ├── common/                # Metrics, logging, configs, reproducibility seeds
│   ├── data/                  # Manifest generation, inventory, transforms
│   ├── explainability/        # Grad-CAM heatmap visualization generator
│   ├── pipeline/              # High-level full-image inference workflows
│   ├── roi/                   # Conjunctiva ROI extraction & quality verification
│   └── segmentation/          # U-Net model, dataset pipeline, mask predictor
│
├── templates/                 # Web interface templates
│   ├── index.html             # Patient screening & Nutritional Guidance UI
│   └── dashboard.html         # Model evaluation and clinical benchmark dashboard
│
├── tests/                     # Unit and integration test suites
│   ├── test_enhanced_features.py
│   ├── test_inference_contract.py
│   ├── test_roi_extraction.py
│   └── test_splits.py
│
└── uploads/                   # Runtime temporary directory for uploaded images
    └── .gitkeep
```

---

## 8. Model Evaluation Metrics

> *The following evaluation metrics are directly verified from the experimental benchmark suite on patient-disjoint clinical cohorts (`artifacts/current/classifier/evaluation_dashboard_data.json`). No metrics are fabricated.*

### Benchmark Overview
- **Dataset:** EYES-DEFY-ANEMIA Clinical Cohort (Italy and India patient cohorts, 202 total subjects).
- **Split Strategy:** Strict patient-disjoint partition (Train: 142, Validation: 31, Test: 29).
- **Calibrated Decision Threshold:** **0.3837** (calibrated on the validation set to maximize clinical sensitivity while maintaining high precision).

### Evaluation Results

| Metric | Validation Set (\(N=31\)) | Independent Test Set (\(N=29\)) |
| :--- | :---: | :---: |
| **Accuracy** | **87.10%** | **79.31%** |
| **Sensitivity (Recall)** | **93.33%** | **92.86%** |
| **Precision** | **82.35%** | **72.22%** |
| **Specificity** | **81.25%** | **66.67%** |
| **F1-Score** | **87.50%** | **81.25%** |
| **ROC-AUC** | **0.8917** | **0.7762** |

### Test Set Confusion Matrix (\(N=29\))
- **True Positives (Anemic detected correctly):** 13
- **True Negatives (Normal detected correctly):** 10
- **False Positives (Normal flagged as at-risk):** 5
- **False Negatives (Anemic missed):** **1** (High sensitivity ensures minimal missed anemia cases)

---

## 9. Installation Instructions

### Prerequisites
- Python 3.10 or 3.11 installed.
- Git installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/vinesh-2005/U-Net-and-EfficientB0-based-anemia-detection-using-dl.git
cd U-Net-and-EfficientB0-based-anemia-detection-using-dl
```

### Step 2: Set Up Virtual Environment
On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
```

On Linux / macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 10. Running the Flask Application

### Method 1: Using the Python CLI
Ensure your virtual environment is active, then run:
```bash
python app.py
```

### Method 2: Using the Windows Launcher Script
Double-click `run_app.bat` or run:
```cmd
run_app.bat
```

Once started, open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Accessing the Model Metrics Dashboard
To view the comprehensive model evaluation and clinical benchmark dashboard, navigate to:
```
http://127.0.0.1:5000/dashboard
```

---

## 11. Deployment to Vercel

The project includes pre-configured [`vercel.json`](vercel.json) and [`.vercelignore`](.vercelignore) for serverless deployment:

1. **Import the repository** into Vercel from your GitHub account.
2. In the **Project Settings** on the Vercel dashboard:
   - Navigate to **Environment Variables**.
   - Add the following environment variable:
     - **Key:** `VERCEL_SUPPORT_LARGE_FUNCTIONS`
     - **Value:** `1`
3. Click **Deploy**. Vercel will build the CPU-optimized serverless function within Fluid Compute limits.
http://127.0.0.1:5000/dashboard
```

---

## 11. Limitations & Future Enhancements

### Current Limitations
- **Image Quality Dependence:** Model predictions rely on sharp, well-lit photographs. Out-of-focus captures or severe motion blur may compromise segmentation accuracy.
- **Illumination Variance:** Extreme color-temperature variations (such as heavily yellow fluorescent or dark ambient lighting) can skew pallor assessment.
- **Dataset Diversity:** While trained on multi-center cohorts (India and Italy), wider multinational validation across diverse demographic groups is recommended prior to clinical deployment.

### Future Enhancements
- **On-Device Edge Deployment:** Porting the TFLite segmentation and classification models into an Android/iOS mobile application for offline field operation.
- **Multi-Class Severity Staging:** Extending binary classification to predict mild, moderate, and severe anemia categories.
- **Real-Time Quality Guidance:** Integrating real-time camera feedback that guides community health workers to capture optimal eyelid positioning before capture.

---

## 12. Medical Disclaimer

> **IMPORTANT MEDICAL NOTICE:**  
> This application is developed strictly for research, educational, and preliminary screening purposes. It **is not a diagnostic medical device** and cannot replace standardized clinical laboratory hemoglobin testing (e.g., automated CBC analyzer or spectrophotometry) or consultation with a qualified medical professional. Anyone suspecting they may have anemia should consult a licensed healthcare practitioner for comprehensive diagnostic evaluation and medically supervised treatment.
