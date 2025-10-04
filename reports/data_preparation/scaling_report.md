# Scaling Report

**Datum/Zeit:** 2025-10-04 11:30:54  
**random_state:** 42  

---

## Overview
- Numerical features scaled using **Min-Max Normalization (0–1)**
- Applied to: `tenure`, `MonthlyCharges`, `TotalCharges`
- Fitted on training data only, applied to validation/test for consistency
- Ensures feature comparability and stability in distance-based and gradient models

---

## Value Ranges (after scaling)
| Feature | Min | Max |
|----------|----:|----:|
| tenure | 0.00 | 1.00 |
| MonthlyCharges | 0.00 | 1.00 |
| TotalCharges | 0.00 | 1.00 |

---

## Output
- `data/scaled/train_scaled.csv`  
- `data/scaled/val_scaled.csv`  
- `data/scaled/test_scaled.csv`

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2

---

## Note
Scaling improves numerical stability, speeds up convergence,  
and prevents bias in algorithms sensitive to feature magnitudes (e.g., XGBoost, kNN, logistic regression).
