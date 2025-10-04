# SMOTE Balancing Report

**Datum/Zeit:** 2025-10-04 11:35:50  
**random_state:** 42  

---

## Overview
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set only  
- Balances class distribution for `Churn` (0 = No, 1 = Yes)  
- Validation and Test sets remain untouched  

---

## Class Distribution

| Split | Class 0 (No Churn) | Class 1 (Churn) |
|--------|-------------------:|----------------:|
| Before SMOTE | 3621 | 1308 |
| After SMOTE (train only) | 3621 | 3621 |

---

## Visualization
*(See corresponding notebook cell for the bar chart of class distribution before and after SMOTE.)*

---

## Output
- `data/balanced/train_balanced.csv`  
- `data/balanced/val_balanced.csv`  
- `data/balanced/test_balanced.csv`  

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2

---

## 🧠 Note
Balancing the training set mitigates bias toward the majority class and helps the model learn minority-class patterns effectively.  
SMOTE synthesizes new samples by interpolating existing minority instances, ensuring a smoother class boundary.
