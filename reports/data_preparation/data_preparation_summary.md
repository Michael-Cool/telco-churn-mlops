# Data Preparation Summary

**Datum/Zeit:** 2025-10-06 08:29:05  
**random_state:** 42  

---

## Overview
All preprocessing steps (Data Selection, Cleaning, Encoding, Scaling, SMOTE) were executed in one reproducible pipeline.  
Intermediate results were stored in `/data/processed/`.

---

## Processed Files
- data/processed/train.csv  
- data/processed/val.csv  
- data/processed/test.csv  

---

## Transformation Overview
| Phase | Transformation | Tools |
|--------|----------------|--------|
| 1 | Stratified Split (70/15/15) | scikit-learn StratifiedShuffleSplit |
| 2 | Cleaning (fix TotalCharges, types) | pandas |
| 3 | Encoding (ordinal + one-hot) | sklearn.preprocessing.OneHotEncoder |
| 4 | Scaling (0–1) | sklearn.preprocessing.MinMaxScaler |
| 5 | Balancing (SMOTE) | imblearn.over_sampling.SMOTE |

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2

---

## Note
This notebook ensures full reproducibility of the Data Preparation phase according to **CRISP-ML(Q)** standards.  
All subsequent modeling and deployment steps should use these processed datasets as input.
