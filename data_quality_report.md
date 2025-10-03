# Data Quality Report – IBM Telco Customer Churn Dataset

## 1. Überblick
- **Datensatz:** IBM Telco Customer Churn Dataset (IBM, 2019)  
- **Größe:** 7.043 Kunden, 21 Variablen  
- **Variablentypen:** 18 kategoriale, 3 numerische  
- **Zielvariable:** Churn (≈ 26,5 %)

---

## 2. Vollständigkeit der Daten
- **Fehlende Werte:** Offiziell keine fehlenden Einträge.  
- **Besonderheit:** `TotalCharges` ist als *object* gespeichert.  
  - In 11 Fällen tritt ein leerer Wert auf (nur bei Kunden mit `tenure = 0`).  
  - Diese Werte sind logisch gleich Null (keine Vertragsmonate → keine Gesamtkosten).  
  - Lösung: Umwandlung in numerischen Datentyp (`float`), fehlende Werte durch `0` ersetzt.

---

## 3. Konsistenz & Datentypen
- **Korrektur:** `TotalCharges` → `float`.  
- **Weitere Variablen:** Typen korrekt (`int` für `SeniorCitizen`, `float` für `MonthlyCharges`).  
- **Kategorische Variablen:** Einheitliche Kodierung in `Yes/No`, Vertragsarten (`Month-to-Month`, `One Year`, `Two Year`), Zahlungsarten etc.  

---

## 4. Verteilungen & Ausreißer
- **Zielvariable:**  
  - Churn-Verteilung: 26,5 % gekündigt, 73,5 % geblieben → **imbalanced data**.  
- **Numerische Variablen:**  
  - **Tenure:** stark rechtsschief, viele Neukunden mit kurzer Vertragsdauer.  
  - **MonthlyCharges:** gleichmäßige Verteilung, Peak bei höheren Gebühren.  
  - **TotalCharges:** korreliert stark mit Vertragsdauer; Ausreißer bei Langzeitkunden mit hohen Summen.  
- **Ausreißerprüfung (Boxplots):** Keine unplausiblen Werte, Ausreißer konsistent mit Geschäftskontext (z. B. hohe TotalCharges bei langjährigen Kunden).  

---

## 5. Plausibilität
- **Demografische Variablen:** Wertebereiche konsistent.  
- **Vertragliche Variablen:** Kategorien plausibel, keine fehlerhaften Einträge.  
- **Zahlungsmethoden:** vier Kategorien, decken die Geschäftspraxis ab.  

---

## 6. Bewertung
- **Datenqualität:** hoch, bis auf die kleine Anpassung bei `TotalCharges`.  
- **Varianz:** ausreichend für Modellbildung (demografisch, vertraglich, finanziell).  
- **Realitätsnähe:** Verteilungen entsprechen plausiblen Mustern im Telco-Sektor.  

---

## 7. Fazit
Der IBM Telco Datensatz ist **vollständig, konsistent und plausibel**.  
Die geringe Anzahl fehlender Werte wurde sachgerecht behandelt.  
Die Datenqualität ist ausreichend für die **Entwicklung und Evaluation** von ML-Modellen im Rahmen dieser Arbeit.

---

## Literatur
IBM. (2019, July 11). *Telco customer churn (11.1.3+)*. IBM Cognos Analytics Community.  
https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113