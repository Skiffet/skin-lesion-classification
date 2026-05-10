# Skin Lesion Classification with Deep Learning
### 204466 Deep Learning — Final Project

**GitHub Repository:** https://github.com/Skiffet/skin-lesion-classification

---

## Project Topic

Multi-class skin lesion classification using a custom Convolutional Neural Network (CNN) trained on the HAM10000 dataset. The model classifies dermoscopy images into 7 types of skin lesions.

---

## Why This Topic?

Skin cancer is one of the most common cancers worldwide. Early and accurate detection significantly improves patient survival rates. However, diagnosis requires experienced dermatologists, which are not always accessible — especially in rural areas.

An automated classification system using deep learning can:
- Assist dermatologists as a second opinion
- Enable early screening in resource-limited settings
- Reduce diagnostic errors caused by human fatigue

This problem is particularly challenging due to:
- High visual similarity between lesion types
- Severe class imbalance in real-world datasets
- Fine-grained features that require pixel-level analysis

---

## Why Deep Learning?

| Approach | Strengths | Weaknesses |
|---|---|---|
| **Traditional ML (Random Forest + HOG)** | Fast training, interpretable, works with small data | Requires manual feature engineering, lower accuracy on complex images |
| **Custom CNN (our model)** | Learns features automatically, high accuracy, handles spatial patterns | Requires large data, longer training time |

Skin lesion classification requires detecting subtle texture, color, and shape patterns at the pixel level. Traditional ML methods rely on hand-crafted features (e.g., HOG) that cannot capture this complexity. CNN automatically learns hierarchical features — from edges to textures to high-level lesion characteristics — making it far more suitable for this problem.

---

## Dataset

**HAM10000** (Human Against Machine with 10000 training images)
- **Source:** Kaggle — [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Total images:** 10,015 dermoscopy images
- **Classes:** 7 types of skin lesions

| Label | Class | Full Name | Count |
|---|---|---|---|
| nv | Melanocytic nevi | Benign mole | ~6,705 |
| mel | Melanoma | Malignant skin cancer | ~1,113 |
| bkl | Benign keratosis | Non-cancerous skin growth | ~1,099 |
| bcc | Basal cell carcinoma | Common skin cancer | ~514 |
| akiec | Actinic keratoses | Pre-cancerous lesion | ~327 |
| vasc | Vascular lesions | Blood vessel abnormality | ~142 |
| df | Dermatofibroma | Benign fibrous nodule | ~115 |

**Split:** 70% Train / 15% Validation / 15% Test (stratified)

---

## CNN Architecture

Custom CNN designed specifically for skin lesion classification:

```
Input: (B, 3, 128, 128)
│
├── ConvBlock 1: Conv(3→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout(0.25)
│   Output: (B, 32, 64, 64)
│
├── ConvBlock 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout(0.25)
│   Output: (B, 64, 32, 32)
│
├── ConvBlock 3: Conv(64→128) → BN → ReLU → Conv(128→128) → BN → ReLU → MaxPool → Dropout(0.25)
│   Output: (B, 128, 16, 16)
│
├── ConvBlock 4: Conv(128→256) → BN → ReLU → Conv(256→256) → BN → ReLU → MaxPool → Dropout(0.40)
│   Output: (B, 256, 8, 8)
│
├── Global Average Pooling → (B, 256)
│
├── FC(256 → 128) → ReLU → Dropout(0.5)
│
└── FC(128 → 7) → Output logits
```

Each ConvBlock contains:
- 2× Conv2d (3×3 kernel, padding=1) — learns spatial features
- BatchNorm2d — stabilizes training
- ReLU — non-linearity
- MaxPool2d (2×2) — reduces spatial dimensions
- Dropout2d — prevents overfitting

---

## Class Imbalance Handling

The dataset is heavily imbalanced (nv has ~58× more samples than df). We address this with two strategies:

1. **WeightedRandomSampler** — minority classes are sampled more frequently during training
2. **Weighted CrossEntropyLoss** — misclassifying rare classes incurs higher penalty

---

## Code Structure

| File | Description |
|---|---|
| `skin_lesion_classification.ipynb` | Main CNN model — data loading, training, evaluation |
| `baseline_random_forest.ipynb` | Random Forest baseline with HOG features |
| `model_comparison.ipynb` | Load results from both models and compare |

### Code Sections (CNN Notebook)

| Section | What it does |
|---|---|
| Setup & Download | Mount Google Drive, download HAM10000 via Kaggle API |
| Data Exploration | Load metadata, visualize class distribution, sample images |
| Dataset & DataLoader | Custom `SkinLesionDataset`, augmentation, `WeightedRandomSampler` |
| Model | `SkinLesionCNN` with 4 ConvBlocks + Global Avg Pool |
| Loss / Optimizer | Weighted CrossEntropyLoss, Adam, ReduceLROnPlateau |
| Training | 30 epochs, saves best model to Google Drive |
| Evaluation | Accuracy, precision, recall, F1, confusion matrix |
| Save Results | Exports `cnn_results.json` for comparison |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Input size | 128 × 128 RGB |
| Batch size | 32 |
| Epochs | 30 |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Loss | Weighted CrossEntropyLoss |
| Framework | PyTorch (Google Colab) |

**Data Augmentation (training only):**
- Random horizontal & vertical flip
- Random rotation ±20°
- Color jitter (brightness, contrast, saturation)
- Normalize with HAM10000 mean/std

---

## Evaluation Results

### Model Comparison

| Metric | Random Forest (HOG) | Custom CNN |
|---|---|---|
| Accuracy | — | — |
| Precision (weighted) | — | — |
| Recall (weighted) | — | — |
| F1 (weighted) | — | — |

*(Results will be filled after running both notebooks on Google Colab)*

---

## References

1. Tschandl, P., Rosendahl, C., & Kittler, H. (2018). **The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.** Scientific Data, 5, 180161. https://doi.org/10.1038/sdata.2018.161

2. Codella, N. et al. (2018). **Skin Lesion Analysis Toward Melanoma Detection: ISIC 2018 Challenge.** arXiv:1902.03368.

3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). **Deep Residual Learning for Image Recognition.** CVPR 2016.

4. Esteva, A. et al. (2017). **Dermatologist-level classification of skin cancer with deep neural networks.** Nature, 542, 115–118.

5. Dalal, N., & Triggs, B. (2005). **Histograms of oriented gradients for human detection.** CVPR 2005. *(HOG features used in Random Forest baseline)*
