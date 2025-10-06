# Feature Importance Report

**Datum/Zeit:** 2025-10-06 10:34:29  
**random_state:** 42  

---

## Overview
Feature importance values were extracted from the final **XGBoost** model  
to understand which variables most strongly influence churn prediction.

---

## Top 10 Features (Quantitative)
| Feature                              |   Importance |
|:-------------------------------------|-------------:|
| Contract                             |    0.25754   |
| InternetService                      |    0.13025   |
| PaymentMethod_Electronic_check       |    0.0719296 |
| OnlineBackup_No_internet_service     |    0.0630912 |
| DeviceProtection_No_internet_service |    0.0584495 |
| tenure                               |    0.0580158 |
| PaperlessBilling_Yes                 |    0.0423229 |
| StreamingMovies_Yes                  |    0.0326658 |
| TechSupport_Yes                      |    0.0305474 |
| MonthlyCharges                       |    0.0279855 |

---

## Interpretation (Qualitative)
- **tenure**: Kunden mit längerer Vertragsdauer kündigen seltener (negativer Einfluss).  
- **MonthlyCharges**: Hohe monatliche Kosten korrelieren positiv mit Abwanderungswahrscheinlichkeit.  
- **Contract_Two year**: Langzeitverträge stabilisieren Kundenbindung.  
- **InternetService_Fiber optic**: Kunden mit Glasfaser zeigen höhere Churn-Tendenz – oft wegen höherer Konkurrenzangebote.  
- **PaymentMethod_Electronic check**: Barrierearme Kündigungsmethoden führen zu höherer Wechselneigung.  
- **OnlineSecurity_No** & **TechSupport_No**: Fehlende Zusatzleistungen gehen häufig mit geringerer Loyalität einher.  
- Weitere Features (z. B. StreamingTV, DeviceProtection) wirken nur schwach, können aber Interaktionseffekte enthalten.

---

## Output
- `reports/modeling/feature_importance_report.md`  
- Feature importance plot generated in `model_training.ipynb`

---

## Reproducibility
- Python 3.13.5  
- pandas 2.3.3  
- scikit-learn 1.7.2  
- xgboost XGBClassifier

---

## References
- Boozary et al. (2025) – *Feature relevance in churn prediction models*  
- Chen & Guestrin (2016) – *XGBoost: A scalable tree boosting system*  
- Studer et al. (2021) – *CRISP-ML(Q): Quality assurance and explainability in ML pipelines*
