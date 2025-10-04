# Encoding Strategy

**Datum/Zeit:** 2025-10-04 10:49:49  
**random_state:** 42  

---

## Overview
- Nominal variables encoded with **One-Hot-Encoding** (drop='first')
- Ordinal variables encoded via **manual mapping**
- Cleaned column names (no spaces or hyphens)
- Handled unknown categories using `handle_unknown="ignore"`

---

## Encoded Columns Summary
**Original columns:** 20  
**After encoding:** 29  

---

## Output
- data/encoded/train_encoded.csv  
- data/encoded/val_encoded.csv  
- data/encoded/test_encoded.csv  

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2
