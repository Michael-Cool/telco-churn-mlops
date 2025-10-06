# GridSearchCV Report

**Datum/Zeit:** 2025-10-06 10:34:28  
**random_state:** 42  

---

## Overview
Hyperparameter optimization was conducted using 5-fold cross-validation (GridSearchCV)  
with ROC AUC as the target metric.

| Model | Best CV ROC AUC | Validation ROC AUC | Train Time (s) | Best Params |
|--------|----------------:|-------------------:|---------------:|-------------:|
| Random Forest | 0.8453 | 0.8308 | 11.78 | {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100} |
| XGBoost | 0.8492 | 0.8359 | 7.57 | {'colsample_bytree': 0.8, 'gamma': 0.1, 'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8} |

---

## Observations
- Random Forest (Breiman, 2001) shows strong baseline robustness; tuning depth & estimators improved stability.  
- XGBoost (Chen & Guestrin, 2016) benefits from moderate learning rates (0.05–0.1) and deeper trees.  
- Validation ROC AUC confirms consistent cross-validation performance.  

---

## Output
- `reports/modeling/gridsearch_report.md`  
- Saved tuned model artifacts in `/models/` (`*_grid.pkl`).  

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2  
- xgboost None (verify version)

---

## References
- Breiman (2001). *Random Forests.* Machine Learning, 45(1).  
- Chen & Guestrin (2016). *XGBoost: A scalable tree boosting system.* KDD '16.  
- Boozary et al. (2025). *Enhancing customer retention with machine learning.*  
- Studer et al. (2021). *CRISP-ML(Q).*  
