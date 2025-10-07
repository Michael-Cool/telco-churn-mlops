# Problemdefinition und Projektziele (Business Value)

## 1. Problemdefinition

Kundenabwanderung (Churn) stellt für Telekommunikationsanbieter ein wesentliches Geschäftsproblem dar.  
Die Gewinnung neuer Kunden ist mit erheblichen Marketing- und Akquisitionskosten verbunden, während der Verlust bestehender Kunden direkte Umsatz- und Ertragsausfälle verursacht.  

Die **Bindung bestehender Kunden** ist deutlich kosteneffizienter als die Neukundengewinnung.  
Reichheld und Sasser (1990) belegen, dass bereits eine Reduktion der Abwanderungsrate um 5 % die Gewinne eines Unternehmens um bis zu 100 % steigern kann, da Bestandskunden im Zeitverlauf profitabler werden. Min et al. zeigen zudem, dass Akquisitionskosten im Telekommunikationssektor im Durchschnitt etwa dreimal so hoch sind wie Retentionskosten und im Wettbewerbsumfeld sogar das Vierfache erreichen können (Min, Zhang, Kim & Srivastava, 2016).
Eine frühzeitige Identifikation von abwanderungsgefährdeten Kunden bedeutet, dass gezielte Retention-Maßnahmen eingeleitet werden können, bevor der Kunde endgültig kündigt.  

Die Herausforderung besteht darin, aus **Vertrags-, Nutzungs- und demografischen Daten** Muster zu erkennen, die auf ein erhöhtes Kündigungsrisiko hindeuten.  
Ziel dieser ARbeit ist ein Ansatz speziell für den **Telco-Sektor in Kombination mit einem AWS-MLOps-Stack**.

### 1.1 Ursachen der Kundenabwanderung im Telco-Sektor

Talaat und Aljadani (2025) identifizieren mehrere zentrale Ursachen für Kundenabwanderung in der Telekommunikationsbranche.  
Eine häufige Ursache ist **mangelnde Servicequalität**, insbesondere in Form von **Verbindungsabbrüchen und Netzstörungen**, die die Kundenzufriedenheit erheblich mindern.  
Ebenso wirken sich **unzureichend gelöste Beschwerden** negativ auf die Kundenbindung aus, da sie das Vertrauen in den Anbieter schwächen.  
Darüber hinaus zeigen sich **Tarif- und Preisaspekte** als treibende Faktoren: Kunden mit **kurzfristigen, nicht vertraglich gebundenen Plänen** wechseln signifikant häufiger den Anbieter als Vertragskunden mit längerer Laufzeit oder Loyalitätsprogrammen.  
Weitere Ursachen liegen in **persönlichen Lebensumständen** (z. B. Umzug) sowie **bewussten Wechselentscheidungen** aufgrund attraktiverer Angebote der Konkurrenz.  
Diese Befunde verdeutlichen, dass Kundenabwanderung nicht nur technisch, sondern auch **verhaltens- und profitabilitätsgetrieben** ist.  
Sie bilden die theoretische Grundlage für das Verständnis der Kundenabwanderung im Telco-Sektor und tragen dazu bei, dass die ML-Anwendung branchenspezifische Muster und Einflussfaktoren realitätsnah abbildet, indem alle im Datensatz enthaltenen Merkmale umfassend analysiert und in die Modellierung einbezogen werden.

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

Talaat, F. M., & Aljadani, A. (2025). *AI-driven churn prediction in subscription services: Addressing economic metrics, data transparency, and customer interdependence.* *Neural Computing and Applications, 37*, 8651–8676. https://doi.org/10.1007/s00521-025-11027-6