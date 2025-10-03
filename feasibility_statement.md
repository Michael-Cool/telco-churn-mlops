# Feasibility Statement

## 1. Datengrundlage
- **Datensatz**: IBM Telco Customer Churn Dataset (IBM, 2019), 7.043 Kunden, 21 Variablen.  
- **Inhalt**: Demografische Merkmale, Vertragsinformationen, Nutzungs- und Zahlungsdaten.  
- **Anonymisierung**: Der Datensatz ist vollständig anonymisiert und enthält keine personenbezogenen Daten.  
- **Varianz**: Hohe Diversität der Features (18 kategoriale, 3 numerische Variablen).  
- **Zielvariable**: Churn ≈ 26,5 % → realistische Abbildung eines Imbalance-Problems.  

## 2. Realistische Anwendbarkeit
- **Branchennähe**: Der synthetische Datensatz ist praxisnah am Telco-Sektor orientiert und wird in Forschung & Lehre häufig als Benchmark eingesetzt.  
- **Bestätigung der Machbarkeit**: Boozary et al. (2025) zeigen, dass ML-Modelle wie XGBoost für Churn Prediction im Telco-Sektor hohe Güte erreichen können.  
- **Datenschutz**: Keine Einschränkungen durch DSGVO, da anonymisierte Daten.  

## 3. Technische Voraussetzungen
- **Format**: CSV (universeller Standard, RFC 4180).  
- **Verarbeitung**: Einfache Integration in Python-Ökosystem (pandas, scikit-learn, XGBoost).  
- **Reproduzierbarkeit**: Datenversionierung via GitHub und MLflow vorgesehen.  
- **Skalierbarkeit**: Problemlos in CI/CD-Workflows, Container (Docker) und AWS-Umgebungen einbindbar.  

## 4. Fazit
Der IBM Telco Customer Churn Datensatz ist **geeignet, realistisch und datenschutzkonform**.  
Er bietet eine valide und praxisnahe Grundlage zur Entwicklung und Evaluation eines skalierbaren ML-Prototyps.  
Die Datenqualität, Varianz und Größe sind ausreichend, um robuste Modelle zu trainieren und die Forschungsfragen dieser Thesis zu beantworten.  

---

## Literatur
IBM. (2019, July 11). *Telco customer churn (11.1.3+).* IBM Cognos Analytics Community. https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113  

Boozary, P., Sheykhan, S., GhorbanTanhaei, H., & Magazzino, C. (2025). Enhancing customer retention with machine learning: A comparative analysis of ensemble models for accurate churn prediction. *Journal of the Japan Institute of Electronics Manufacturing and Intelligent Engineering, 14*(3), Article 100331. https://doi.org/10.1016/j.jjimei.2025.100331  

Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., & Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. *Machine Learning and Knowledge Extraction, 3*(2), 392–413. https://doi.org/10.3390/make3020020