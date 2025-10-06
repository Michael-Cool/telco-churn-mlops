# Model Performance Analysis

**Datum/Zeit:** 2025-10-06 15:43:27  
**random_state:** 42  

---

## Overview
Evaluation of the final **XGBoost** model on the independent test dataset.  
Metrics reflect overall generalization capability and robustness.

| Metric | Score |
|:--------|------:|
| Accuracy | 0.800 |
| Precision | 0.658 |
| Recall | 0.514 |
| F1-Score | 0.577 |
| ROC AUC | 0.854 |
| Log Loss | 0.406 |
| MCC | 0.455 |

---

## Visualizations
- Confusion Matrix
- ROC Curve
- Precision–Recall Curve

---

## Interpretation
- **Bias vs. Variance:** Model shows stable generalization with balanced recall and precision.  
- **Overfitting check:** Test ROC AUC close to validation value (~0.85) → No significant overfitting.  
- **Stability:** Consistent with cross-validation results from tuning phase.  

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2  
- xgboost XGBClassifier  

---

## References
- Boozary et al. (2025) – *Enhancing customer retention with ML*  
- Chen & Guestrin (2016) – *XGBoost: A scalable tree boosting system*  
- Studer et al. (2021) – *CRISP-ML(Q)*  
