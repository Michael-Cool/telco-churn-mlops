# Data Selection

**Datum/Zeit:** 2025-10-04  
**random_state:** 42  
**Quelle:** data/raw/IBMTelco_Datensatz.csv  

---

## Umfang

| Split | n | Churn n | Churn % |
|-------|---:|---:|---:|
| Gesamt | 7043 | 1869 | 26.5 % |
| Train | 4929 | 1308 | 26.5 % |
| Validation | 1057 | 281 | 26.6 % |
| Test | 1057 | 280 | 26.5 % |

---

## Vorgehen

- Alle Variablen außer `customerID` wurden berücksichtigt.  
- Der Datensatz wurde **stratifiziert** nach der Zielvariable `Churn` in drei Splits aufgeteilt:  
  - 70 % Training  
  - 15 % Validation  
  - 15 % Test  
- Dadurch bleibt die Klassenverteilung in allen Teilmengen konsistent.  
- Reproduzierbarkeit wurde durch `random_state = 42` sichergestellt.  
- Aufteilung mittels `StratifiedShuffleSplit` aus `scikit-learn`.  

---

## Begründung nach CRISP-ML(Q)

Die stratifizierte Stichprobenziehung stellt sicher, dass  
die Zielvariable in allen Splits repräsentativ bleibt und  
somit die **Modellvalidierung unter gleichen Bedingungen** erfolgt.  

Die Dokumentation des Seeds und der Bibliotheksversionen  
gewährleistet die **Reproduzierbarkeit** der Ergebnisse –  
ein zentrales Qualitätsmerkmal nach *Studer et al. (2021)*.

---

## Reproduzierbarkeit

| Komponente | Version |
|-------------|----------|
| Python | 3.11 |
| pandas | 2.2 |
| scikit-learn | 1.5 |
| Betriebssystem | macOS |

---

## Output-Dateien

- `data/splits/train.csv`  
- `data/splits/val.csv`  
- `data/splits/test.csv`  

Diese Splits werden in allen weiteren Phasen  
(**Data Cleaning, Encoding, Modeling, Evaluation**) unverändert verwendet.