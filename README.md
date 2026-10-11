# Multi-Modal Learning for Inpatient Length of Stay Prediction

Predicts hospital length of stay (LOS) by fusing three data modalities from the MIMIC-IV clinical database: structured EHR features, chest X-ray image embeddings, and radiology report text embeddings.

Google Colab: [Open Notebook](https://colab.research.google.com/drive/1N7ht4O0pkReu48mhRsgAa4ShZMLYSA1S)

---

## Overview

Length of stay (LOS) — the time between hospital admission and discharge — is one of the strongest predictors of hospital resource utilization. Accurate LOS forecasting supports capacity planning, early discharge coordination, and care management.

This project replicates and extends the deep fusion architecture from Chen et al. (2024) using a modular, interpretable pipeline built on MIMIC-IV v3.1 and MIMIC-CXR.

---

## Dataset

| Split | Count |
|---|---|
| Source admissions (MIMIC-IV v3.1) | 534,148 |
| CXR-linked admissions | 36,595 |
| Complete multimodal cases (final cohort) | **398** |
| Feature columns | **772** |

**Data sources:**
- [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) — structured EHR (admissions, diagnoses, procedures, labs, ICU stays)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) — chest X-ray DICOM images and radiology reports

> Access to both datasets requires a PhysioNet credentialed account and completion of required training.

---

## Pipeline Architecture

**Three parallel modality encoders feed into a shared XGBoost fusion model:**

```
Chest X-ray Images      → ResNet18 → Embeddings (512-d) → PCA (32-d)  ─┐
Radiology Text Reports  → LSTM     → Embeddings (64-d)  → PCA (16-d)  ─┤→ Concatenation → XGBoost → Predicted LOS
Structured EHR Features → Feature Engineering → XGBoost (base learner) ─┘
```

### Image Encoder
- Pre-trained **ResNet18** (ImageNet weights) as feature extractor
- Final classification layer removed; 512-d embeddings extracted
- Regressor head: Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1)
- Standalone image model: RMSE 7.92, MAE 5.64
- PCA to 32 components applied before fusion

### Text Encoder
- Vocabulary: 10,000 tokens, max sequence length 200
- **LSTM** with embedding dim 128, hidden dim 64
- Regressor head: Linear(64→32) → ReLU → Linear(32→1)
- Standalone text model: RMSE 10.11
- PCA to 16 components applied before fusion

### Structured EHR Encoder
Features engineered from MIMIC-IV hospital and ICU tables:
- **Demographics**: age, gender, admission type/location, insurance, marital status, race
- **Diagnosis categories**: 15 ICD-mapped disease groups (Circulatory, Respiratory, Digestive, Endocrine, etc.)
- **Procedure categories**: ICD-9/10 mapped surgical procedure counts
- **Laboratory values**: 10 key lab items (Sodium, Creatinine, Hemoglobin, etc.) aggregated as mean/max/min/first per admission
- **ICU features**: number of ICU stays, total ICU days, first/last care unit
- **Comorbidity flags**: binary indicators for diabetes, hypertension, CKD, heart failure, COPD, cancer
- **Temporal features**: admission weekday, weekend flag, admission hour/month
- Standalone structured model (XGBoost, 200 estimators): RMSE 3.77, MAE 1.59

### Fusion
- All three modality embeddings concatenated (structured + 32-d image PCA + 16-d text PCA = 772 total features)
- Final **XGBoost** fusion model: 400 estimators, max_depth=6, learning_rate=0.05

---

## Results

| Metric | Paper (Chen et al. 2024) | This Project | Difference |
|--------|--------------------------|--------------|------------|
| MAE | 3.868 | 5.039 | +30% |
| RMSE | 5.531 | 7.140 | +29% |
| R² | 0.604 | 0.172 | −72% |
| EVAR | 0.604 | 0.173 | −71% |

**Multimodal fusion (final model): MAE 5.04, RMSE 7.14, R² 0.172**

The performance gap relative to the paper is primarily attributable to dataset size: the final multimodal cohort contains only 398 complete-case admissions (requiring all three modalities to align), whereas the original paper used a far larger training set with end-to-end joint training. The takeaway is that multimodal fusion is highly sensitive to data completeness — with limited samples, performance is constrained regardless of model complexity, and simpler interpretable models remain competitive.

---

## Key Design Contributions

- **Modular pipeline** enables clear separation of modality contributions and interpretability of structured vs. multimodal performance
- **Doctor-informed feature engineering**: peak lab values (max/min) emphasized over averages; comorbidity flags and ICU exposure features added per clinical feedback
- **PCA dimensionality reduction** on neural embeddings reduced overfitting in high-dimensional space and improved fusion stability
- **CXR linking**: matched X-rays to admissions using StudyDate + StudyTime within admission window; filtered to AP/PA views; selected first CXR per admission

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | MIMIC-IV v3.1, MIMIC-CXR (PhysioNet) |
| EHR Processing | pandas, numpy |
| Image Model | PyTorch, torchvision (ResNet18) |
| Text Model | PyTorch (LSTM), custom vocabulary |
| Structured + Fusion | XGBoost, scikit-learn |
| Dimensionality Reduction | scikit-learn PCA |
| Environment | Google Colab (GPU) |

---

## Repository Contents

```
mimic-los-prediction/
├── ISE_543_Final_Codes.ipynb    # Full pipeline notebook (Google Colab)
├── ISE_543_Group_4_Final_PPT.pdf  # Project presentation slides
└── README.md
```

> The MIMIC datasets are not included (PhysioNet credentialed access required). The notebook references data paths under `/scratch1/hkavuri/mimic_project_data/` as used on the USC HPC cluster.

---

## Course

**ISE 543: Data Analytics Consulting** — Spring 2026  
USC Viterbi School of Engineering  
Instructor: Prof. Maryam Pishgar

---

## Author

Himaja Kavuri — MS Analytics, USC Viterbi School of Engineering  
Portfolio: [himajakavuri23.github.io](https://himajakavuri23.github.io)  
LinkedIn: [linkedin.com/in/himaja-kavuri](https://linkedin.com/in/himaja-kavuri)
