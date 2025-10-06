# Model Serialization Log

**Datum/Zeit:** 2025-10-06 10:39:54  
**random_state:** 42  

---

## Serialized Models
| Datei | Beschreibung |
|--------|---------------|
| `random_forest_model.pkl` | Tuned Random Forest model (Baseline + GridSearch) |
| `xgboost_model.pkl` | Tuned XGBoost model (Final candidate) |
| `final_model.pkl` | Best performing model retrained on full Train + Val dataset |

---

## Reproducibility Information
| Komponente | Version |
|-------------|----------|
| Python | 3.13.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| xgboost | XGBClassifier |
| joblib | 1.5.2 |

---

## Notes
- All models serialized using **joblib** for compatibility with scikit-learn and FastAPI deployment.  
- Random seed fixed (`random_state = 42`) to ensure deterministic results.  
- Stored under `/models/` and tracked via Git for versioning.  

---

## References
- Studer et al. (2021) – *CRISP-ML(Q): Quality assurance and reproducibility in ML pipelines*  
- Chikkala et al. (2025) – *Automating MLOps with GitHub Actions and Azure*  
- Boozary et al. (2025) – *Enhancing customer retention with machine learning*
