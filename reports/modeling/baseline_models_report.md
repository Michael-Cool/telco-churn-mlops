# Baseline Models Report

**Datum/Zeit:** 2025-10-06 10:34:09  
**random_state:** 42  

---

## Overview
Both baseline models were trained with default parameters using the processed Telco dataset.

| Model | Accuracy | F1 | ROC AUC | Train Time (s) |
|--------|----------:|---:|---:|---------------:|
Random Forest 0.779 0.517 0.814 0.12
      XGBoost 0.770 0.532 0.808 0.11

---

## Observations
- Random Forest (Breiman, 2001) provides a solid benchmark for ensemble performance.  
- XGBoost (Chen & Guestrin, 2016) typically achieves higher ROC-AUC due to gradient boosting.  
- Differences in training time and scores will guide the upcoming hyperparameter tuning (GridSearchCV).  

---

## Output
- `reports/modeling/baseline_models_report.md`  
- Model objects currently stored in memory (not serialized yet).  

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2

---

## References
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.  
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD '16.  
- Boozary et al. (2025). *Enhancing customer retention with machine learning.*  
- Studer et al. (2021). *CRISP-ML(Q): A machine learning process model with quality assurance methodology.*
