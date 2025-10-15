# Data Requirements – IBM Telco Customer Churn Dataset

## 1. Ziel
Die Definition der Data Requirements legt fest, welche Variablen für die Churn-Prediction benötigt werden, welche Transformationen erforderlich sind und wie mit Datenungleichgewichten umgegangen wird.  
Dies bildet die Brücke zwischen **Data Understanding** und **Data Preparation** im CRISP-ML(Q)-Prozess.

---

## 2. Relevante Variablen
Der IBM Telco Customer Churn Datensatz enthält 21 Variablen (7043 Kunden).  
Die Zielvariable ist **Churn** (Yes/No).

- **Demografische Merkmale**
  - `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Vertragsmerkmale**
  - `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Service-Nutzung**
  - `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,  
    `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Finanzielle Variablen**
  - `MonthlyCharges`, `TotalCharges`, `tenure`
- **Nicht relevant**
  - `customerID` (identifiziert Kunden, hat keinen prädiktiven Mehrwert)

---

## 3. Transformationen
Zur Modellierung sind die folgenden Transformationen notwendig:

- **One-Hot-Encoding**  
  Für nominale kategoriale Variablen (`gender`, `Contract`, `PaymentMethod`, `InternetService`, …).

- **Ordinal-Encoding**  
  Für Variablen mit inhärenter Ordnung (z. B. `Contract`: Month-to-Month = 1, One Year = 12, Two Year = 24).

- **Numerische Skalierung**  
  - `MonthlyCharges`, `TotalCharges`, `tenure` werden mit **Min-Max-Normalisierung** skaliert, um unterschiedliche Wertebereiche vergleichbar zu machen (vgl. Boozary et al., 2025).

- **Fehlende Werte**  
  - `TotalCharges`: 11 fehlende Einträge → werden zu `0` gesetzt, da Kunden `tenure = 0` haben.

---

## 4. Umgang mit Klassenungleichgewicht
Die Zielvariable **Churn** ist ungleich verteilt (≈ 26,5 % Churn vs. 73,5 % Non-Churn).  
- **Problem:** Modelle neigen dazu, die Mehrheitsklasse (Nicht-Kündiger) zu bevorzugen.  
- **Lösung:** Einsatz des **SMOTE-Verfahrens** (*Synthetic Minority Over-sampling Technique*) auf dem Trainingsdatensatz, um synthetische Beispiele für die Minderheitsklasse zu generieren (Chawla et al., 2002).

---

## 5. Datenaufteilung
Für Training und Evaluation werden die Daten aufgeteilt in:
- **Training:** 70 %  
- **Validation:** 15 %  
- **Test:** 15 % (unberührt während des Trainingsprozesses)  

Der Split erfolgt mit **festem Random Seed**, um Reproduzierbarkeit zu gewährleisten.

---

## 6. Plausibilitätsregeln und Wertebereiche

Zur Sicherstellung der Datenqualität wurden für den IBM Telco Customer Churn Datensatz Plausibilitätsregeln und Wertebereiche definiert.  
Diese Regeln dienen dazu, unplausible oder fehlerhafte Werte frühzeitig zu erkennen und auszuschließen.  
Sie basieren auf den in der EDA gewonnenen Erkenntnissen sowie branchenspezifischen Annahmen aus der Telekommunikation.

| Variable | Typ | Gültige Werte / Bereich | Begründung |
|-----------|-----|-------------------------|-------------|
| `MonthlyCharges` | numerisch | > 0 und ≤ 120 | Negative oder unrealistisch hohe Gebühren ausgeschlossen |
| `TotalCharges` | numerisch | ≥ 0 | Negative Gesamtkosten sind nicht möglich |
| `tenure` | numerisch | 0–72 | Maximale Vertragslaufzeit beträgt 72 Monate (6 Jahre) |
| `Contract` | kategorial | {Month-to-Month, One year, Two year} | Gültige Vertragsarten laut Datendefinition |
| `PaymentMethod` | kategorial | {Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)} | Mögliche Zahlungsarten gemäß Originaldaten |
| `InternetService` | kategorial | {DSL, Fiber optic, No} | Erlaubte Arten von Internetdiensten |
| `Churn` | binär | {Yes, No} | Zielvariable (Kunde kündigt / bleibt) |
| `SeniorCitizen` | binär | {0, 1} | 1 = Senior, 0 = kein Senior |
| `gender` | kategorial | {Male, Female} | laut Originaldaten |
| `Dependents` | kategorial | {Yes, No} | Abhängige Personen vorhanden oder nicht |
| `Partner` | kategorial | {Yes, No} | Partnerstatus des Kunden |
| `PaperlessBilling` | kategorial | {Yes, No} | Nutzung papierloser Abrechnung |

**Hinweis:**  
Diese Regeln wurden im Rahmen der Datenbereinigung überprüft (vgl. `reports/data_preparation/data_cleaning_report.md`).  
Alle Variablen erfüllten die definierten Bedingungen; es waren keine manuellen Korrekturen erforderlich.

---

## 7. Fazit
- Der IBM Telco Datensatz bietet ausreichende Variablenvielfalt für eine robuste Churn-Prediction.  
- Transformationen (Encoding, Skalierung, Bereinigung) stellen sicher, dass die Daten in ML-Modelle (z. B. XGBoost, Random Forest) integriert werden können.  
- Durch SMOTE und sauberen Datensplit wird die Basis für faire und reproduzierbare Modellvergleiche gelegt.  

---

## Literatur
Boozary, P., Sheykhan, S., GhorbanTanhaei, H., & Magazzino, C. (2025). *Enhancing customer retention with machine learning: A comparative analysis of ensemble models for accurate churn prediction*. Journal of the Japan Institute of Electronics Manufacturing and Intelligent Engineering, 14(3), Article 100331. https://doi.org/10.1016/j.jjimei.2025.100331  

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique*. Journal of Artificial Intelligence Research, 16, 321–357. https://doi.org/10.1613/jair.953  

IBM. (2019, July 11). *Telco customer churn (11.1.3+)*. IBM Cognos Analytics Community.  
https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113