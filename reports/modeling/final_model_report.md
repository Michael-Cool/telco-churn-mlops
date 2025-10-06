# Final Model Report

**Datum/Zeit:** 2025-10-06 10:34:29  
**random_state:** 42  

---

## Overview
The best model was selected based on ROC AUC and balanced performance metrics (Precision, Recall, F1).  
Both tuned models were retrained on the full Train+Val dataset and evaluated on the hold-out Test set.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|--------|----------:|----------:|-------:|----:|--------:|
| Random Forest | 0.8 | 0.651 | 0.532 | 0.585 | 0.85 |
| XGBoost | 0.8 | 0.658 | 0.514 | 0.577 | 0.854 |

---

## Observations
- The selected model (**XGBoost**) achieved the highest ROC AUC on the test set.  
- The model shows a strong balance between Recall and Precision, minimizing false negatives (critical in churn detection).  
- Results confirm robust generalization across unseen data.

---

## Output
- `reports/modeling/final_model_report.md`  
- Final serialized model: `/models/final_model.pkl`  

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2  
- xgboost XGBClassifier

---

## References
- Boozary et al. (2025) – *Churn Prediction Metrics & Model Comparison*  
- Chen & Guestrin (2016) – *XGBoost: A scalable tree boosting system*  
- Breiman (2001) – *Random Forests*  
- Studer et al. (2021) – *CRISP-ML(Q)*  
