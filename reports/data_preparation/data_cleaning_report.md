# Data Cleaning Report

**Datum/Zeit:** 2025-10-04 09:55:23  
**random_state:** 42  

---

## Overview
- Missing values handled (TotalCharges = 0 where tenure = 0)
- Data types standardized:
  - SeniorCitizen → int (0/1)
  - All other categorical fields → string
- Outliers checked (Boxplot of tenure, MonthlyCharges, TotalCharges)
- Consistency verified: no negative or non-numeric values

---

## Missing Values (before cleaning)
|                  |   0 |
|:-----------------|----:|
| gender           |   0 |
| SeniorCitizen    |   0 |
| Partner          |   0 |
| Dependents       |   0 |
| tenure           |   0 |
| PhoneService     |   0 |
| MultipleLines    |   0 |
| InternetService  |   0 |
| OnlineSecurity   |   0 |
| OnlineBackup     |   0 |
| DeviceProtection |   0 |
| TechSupport      |   0 |
| StreamingTV      |   0 |
| StreamingMovies  |   0 |
| Contract         |   0 |
| PaperlessBilling |   0 |
| PaymentMethod    |   0 |
| MonthlyCharges   |   0 |
| TotalCharges     |   0 |
| Churn            |   0 |
---

## Consistency Checks
| Check | Count |
|--------|-------:|
| Negative tenure | 0 |
| Negative charges | 0 |
| Special characters | 3066 |

---

## Output
- data/cleaned/train_clean.csv  
- data/cleaned/val_clean.csv  
- data/cleaned/test_clean.csv  

---

## Note
All transformations follow reproducibility guidelines according to **CRISP-ML(Q)** (Studer et al., 2021).
