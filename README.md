# Classifying Retinal Images for Diabetic Retinopathy using Vision Transformers

This project develops and evaluates deep learning models for automated classification of retinal fundus images to detect Diabetic Retinopathy (DR). It compares a simple PyTorch CNN baseline against a fine‑tuned [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224), with emphasis on generalization and minority‑class performance.

## Key Features

* **DR Classification:** Automated detection of diabetic retinopathy from retinal fundus images.  
* **Comparative Model Analysis:** Simple PyTorch CNN baseline vs fine‑tuned ViT (Hugging Face) for classification.  
* **Targeted Fine‑tuning:** Head‑only fine‑tuning of pre‑trained ViT with a custom 2‑class classifier for sample‑efficient adaptation.  
* **Robust Preprocessing & Augmentation:** PyTorch + torchvision transforms including retinal center‑crop and CLAHE‑style contrast normalization; resizing, normalization, and batched DataLoaders.  
* **Minority‑class Focus:** Specific attention to improving detection rates for the non‑DR class in an imbalanced dataset.  
* **Detailed Performance Evaluation:** Thorough assessment using key metrics including Accuracy, Precision, Recall, and F1-score.

## Performance

Observed performance:

* **CNN Baseline:** Achieved **58% accuracy**.  
* **Fine-tuned Vision Transformer (ViT):** Achieved **81% accuracy**, representing a **23% performance improvement** over the CNN baseline.  
* **Minority‑class generalization (non‑DR):**  
  * Precision increased from **0.24 to 0.69**.  
  * Recall increased from **0.12 to 0.74**.  
* **Error profile:** Reduced false negatives and false positives.

## Tech Stack

* **Deep Learning:** PyTorch, torchvision, Hugging Face Transformers  
* **Model Architectures:** CNN baseline, [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224)  
* **ML/Analytics:** scikit‑learn, NumPy, pandas, Matplotlib, Seaborn  
* **Dataset:** [IDRiD (Indian Diabetic Retinopathy Image Dataset)](https://idrid.grand-challenge.org/Data/)

## System Overview

### Architecture
- Data pipeline in PyTorch with `DataLoader` batching and device‑aware execution (CPU/GPU).  
- CNN baseline trained end‑to‑end; ViT fine‑tuned by freezing the backbone and training a 2‑class linear head.  
- Deterministic runs with `torch.manual_seed(42)` and controlled transforms for comparability.

### Data & Preprocessing
- Source: IDRiD retinal fundus images with CSV labels for train/val/test.  
- Transform stack: retinal center‑crop, CLAHE‑style contrast enhancement, resizing to ViT input, normalization; train‑time augmentation for robustness.  
- Class imbalance addressed via weighted loss for CNN; evaluation emphasizes minority‑class metrics.

### Training & Evaluation
- Loss: Cross‑entropy (ViT); weighted BCE/Cross‑entropy for CNN baseline.  
- Optimizer: Adam; per‑epoch tracking of loss and accuracy on train/val splits.  
- Test protocol: held‑out test set evaluation with accuracy and per‑class precision/recall.

### Results (replicable)
- ViT test accuracy: **81%**; CNN baseline: **58%** (**+23 pp**).  
- Minority‑class improvements: precision **0.24→0.69**, recall **0.12→0.74**.  
- Learning‑curve analysis used to diagnose overfitting and tune epochs/regularization.

### Reproducibility & Ops
- Versioned dependencies in `requirements.txt`.  
- Self‑contained notebook pipeline (data loading → transforms → training → evaluation).  
- Clear separation of train/val/test and consistent preprocessing across splits.

### Limitations & Next Steps
- Expand training data and explore stronger augmentation/mixup for class imbalance.  
- Unfreeze selective ViT blocks with discriminative learning rates for higher ceilings.  
- Add calibration (ECE), ROC‑AUC, and threshold tuning to minimize clinical risk.  
- Export ONNX/TorchScript and add runtime validation for deployment‑readiness.

## How to Run

1. Set Up Environment:

```bash
python \-m venv dr\_env  
source dr\_env/bin/activate  
```

2. **Install Dependencies:**

```bash
pip install \-r requirements.txt
```

3. Download Dataset:  
   Obtain the IDRiD dataset from its official source (often requires registration or agreement to terms of use). Place the images and labels in a designated data/ directory.  
4. Run Jupyter Notebook/Scripts:  
   Execute the provided Jupyter notebooks or Python scripts to preprocess data, train models, and evaluate performance.

## Dataset

This project primarily utilizes the [**IDRiD (Indian Diabetic Retinopathy Image Dataset)**](https://idrid.grand-challenge.org/Data/), a publicly available dataset specifically curated for diabetic retinopathy detection and grading. It provides retinal fundus images along with corresponding ground truth labels for DR severity.

## Future Work

Future work could explore:

* Utilizing different model architectures or more specialized pre-trained vision transformers.  
* Implementing advanced dataset balancing techniques.  
* Collecting and incorporating more diverse data.  
* Applying more rigorous testing and validation strategies.

