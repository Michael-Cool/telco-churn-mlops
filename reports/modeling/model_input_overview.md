# Model Input Overview

**Datum/Zeit:** 2025-10-06 10:34:08  
**random_state:** 42  

---

## Datasets

| Split | Samples | Features | Churn % |
|-------|----------|-----------|----------|
| Train | 4,929 | 29 | 26.5% |
| Val | 1,057 | 29 | 26.6% |
| Test | 1,057 | 29 | 26.5% |

---

## Checks

- Same columns across all splits: True
- Missing values (train/val/test): 0/0/0
- Balanced training set confirmed (SMOTE applied).

---

## Notes

- Features are fully numeric, scaled (0–1), and clean.
- Target variable: `Churn` (0 = No, 1 = Yes).
- These datasets serve as input for all subsequent modeling steps (Random Forest, XGBoost, GridSearchCV).
- All transformations follow reproducibility standards per **CRISP-ML(Q)** (Studer et al., 2021).

---

## References

- Boozary et al. (2025): *Enhancing customer retention with machine learning* – Defines modeling baseline and metrics.  
- Studer et al. (2021): *CRISP-ML(Q)* – Emphasizes reproducibility and data traceability.
