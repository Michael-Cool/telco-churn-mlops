# Problemdefinition und Projektziele (Business Value)

## 1. Problemdefinition

Kundenabwanderung (Churn) stellt für Telekommunikationsanbieter ein wesentliches Geschäftsproblem dar.  
Die Gewinnung neuer Kunden ist mit erheblichen Marketing- und Akquisitionskosten verbunden, während der Verlust bestehender Kunden direkte Umsatz- und Ertragsausfälle verursacht.  

Die **Bindung bestehender Kunden** ist deutlich kosteneffizienter ist als die Neukundengewinnung.  
Reichheld und Sasser (1990) belegen, dass bereits eine Reduktion der Abwanderungsrate um 5 % die Gewinne eines Unternehmens um bis zu 100 % steigern kann, da Bestandskunden im Zeitverlauf profitabler werden. Min et al. zeigen zudem, dass Akquisitionskosten im Telekommunikationssektor im Durchschnitt etwa dreimal so hoch sind wie Retentionskosten und im Wettbewerbsumfeld sogar das Vierfache erreichen können (Min, Zhang, Kim & Srivastava, 2016).
Eine frühzeitige Identifikation von abwanderungsgefährdeten Kunden bedeutet, dass gezielte Retention-Maßnahmen eingeleitet werden können, bevor der Kunde endgültig kündigt.  

Die Herausforderung besteht darin, aus **komplexen Vertrags-, Nutzungs- und demografischen Daten** Muster zu erkennen, die auf ein erhöhtes Kündigungsrisiko hindeuten.  
Bisherige Arbeiten zu Churn Prediction im Machine Learning adressieren entweder die Modellgüte (Boozary et al., 2025) oder die technische Infrastruktur (Woźniak et al., 2025; Chikkala et al., 2025), jedoch fehlt ein praxisnaher, übertragbarer Ansatz speziell für den **Telco-Sektor in Kombination mit einem AWS-MLOps-Stack**.

---

## 2. Projektziele

### 2.1 Business-Ziele
- **Frühzeitige Erkennung abwanderungsgefährdeter Kunden** auf Basis des IBM Telco Customer Churn Datensatzes (7.043 Kunden, 21 Variablen).  
- **Bereitstellung fundierter Entscheidungsgrundlagen** für Retention-Maßnahmen, um die Churn-Rate als zentrales Unternehmensziel zu senken.  
- **Stärkung der Kundenbindung** durch datengetriebene, proaktive Interventionen.  

### 2.2 Technische Ziele
- Entwicklung einer **skalierbaren ML-API** zur Vorhersage von Kundenabwanderung.  
- Umsetzung einer **produktnahen MLOps-Architektur** mit FastAPI, Docker, GitHub Actions (CI/CD) und AWS (EC2, S3).  
- Sicherstellung von **Reproduzierbarkeit, Skalierbarkeit und Wartbarkeit** durch den Einsatz von Versionierung, automatisierten Pipelines und Monitoring.  
- Evaluation der Modellgüte mit etablierten Metriken (Accuracy, Precision, Recall, F1, ROC AUC) sowie ergänzenden Kriterien (Robustheit, Erklärbarkeit, Ressourcenbedarf).

### 2.3 Wissenschaftlicher Mehrwert
- Übertragung von bestehenden Forschungsergebnissen (z. B. Boozary et al., Chikkala et al., Woźniak et al.) auf die Telco-Domäne mit AWS-Technologie.  
- Dokumentation eines **vollständigen End-to-End-Workflows** entlang des CRISP-ML(Q)-Prozessmodells.  
- Erstellung eines **reproduzierbaren Prototyps**, der als Blaupause für ähnliche Projekte in datengetriebenen Branchen dienen kann.

---

## 3. Erwarteter Business Value

- **Wirtschaftlich**: Reduktion der Kosten durch gezielte Kundenbindung und Senkung der Abwanderungsrate.  
- **Operativ**: Schnell einsetzbarer, reproduzierbarer Workflow, der den Einsatz von Machine Learning im Telco-Kontext beschleunigt.  
- **Forschungs- und Entwicklungsnutzen**: Wissenschaftlich fundierte Dokumentation eines MLOps-Setups mit Praxisbezug.  

Damit schließt dieses Projekt eine Lücke zwischen **rein akademischen Modellvergleichen** und **praxisnahen MLOps-Architekturen** und liefert einen konkreten Mehrwert für Telco-Unternehmen wie auch für Entwickler und MLOps-Praktiker.




## Literatur
Reichheld, F. F., & Sasser, W. E. (1990). *Zero defections: Quality comes to services*. *Harvard Business Review, 68*(5), 105–111. Retrieved October 3, 2025, from https://hbr.org/1990/09/zero-defections-quality-comes-to-services

Min, S., Zhang, X., Kim, N., & Srivastava, R. K. (2016). Customer acquisition and retention spending: An analytical model and empirical investigation in wireless telecommunications markets. Journal of Marketing Research, 53(5), 728–744. https://doi.org/10.1509/jmr.14.0170