# Handoff — ML4B Gym Exercise Recognition

> **Zweck.** Knappe Zusammenfassung für ein Team, das unser Projekt verstehen und
> **vorstellen** soll. Sie erklärt, worum es geht, was die App kann, wie wir
> vorgegangen sind, welche Entscheidungen wir getroffen haben und was wir gelernt
> haben. Tiefe Details stehen in `docs/DECISIONS.md` und den CRISP-DM-Notebooks —
> dieses Dokument ist der Einstieg.

---

## 1. Worum geht es?

Wir erkennen **Fitnessstudio-Übungen aus Apple-Watch-Bewegungsdaten**. Die Uhr
misst am Handgelenk Beschleunigung und Drehrate. Man nimmt ein Workout mit der
kostenlosen App **Sensor Logger** auf, lädt die Datei in unsere
**Streamlit-Web-App**, und bekommt pro 2-Sekunden-Fenster die erkannte Übung mit
Konfidenz.

**Erkannte Übungen (3):** Bizeps-Curl, Trizeps-Extension, Rudern (`row`).

## 2. Was kann die App?

- CSV-/ZIP-Upload aus Sensor Logger (Apple Watch).
- Vorverarbeitung mit **exakt demselben Code** wie im Training (kein Duplizieren).
- **Zeitstrahl**: welche Übung wann, plus Konfidenz pro Fenster.
- **Pausen** zwischen Sätzen werden automatisch als `rest` markiert.
- Unsichere Fenster → `uncertain`; unbekannte Bewegungen → `unknown` (ehrliches
  „weiß ich nicht" statt Raten).
- **„Detected Sets"**: Fenster werden zu Sätzen (Bouts) zusammengefasst — Nutzer
  denken in Sätzen, nicht in 2-s-Fenstern.
- **Zwei-Modell-Vergleich**: Baseline (nur Kaggle) vs. aktuelles Modell (Kaggle +
  eigene Daten) laufen parallel auf derselben Aufnahme — der Effekt eigener Daten
  wird sichtbar.
- **Performance-Seite**: Accuracy, Confusion-Matrix, F1 pro Klasse.
- Start mit **einem** Befehl: `uv run streamlit run app/streamlit_app.py` (bzw.
  `make run`).

## 3. Wie sind wir vorgegangen? (CRISP-DM)

Business → Data Understanding → Data Preparation → Modeling → Evaluation →
Deployment. Eine **gemeinsame Pipeline** verbindet alles; der Kern-Grundsatz lautet:
Trainings- und App-Pipeline nutzen **identischen** Vorverarbeitungscode.

```
Sensor Logger CSV → preprocessing → features → Random Forest → Streamlit-App
```

## 4. Welche Modelle haben wir abgewogen — und warum Random Forest?

- **scikit-learn statt Deep Learning.** Jedes Fenster wird zu einem festen
  Merkmalsvektor → klassische tabellarische Klassifikation. Deep Learning (CNN/LSTM)
  wäre für handgebaute Features und ein gemischt erfahrenes Team Overkill.
- **Random Forest vs. XGBoost vs. SVM-RBF**, verglichen über **macro-F1**:
  - **Random Forest (gewählt):** bester F1, schnell, nativ mehrklassig, gut
    interpretierbar (`feature_importances_`).
  - **XGBoost:** als dokumentiertes Backup behalten.
  - **SVM-RBF:** verworfen (schlechterer F1, langsame Wahrscheinlichkeits-Kalibrierung).
- **Bewusst gedämpft (`max_depth=20`).** Ungebremst memorierten die Bäume die
  Trainingsdaten; das Deckeln kostet kaum F1, liefert aber **ehrlichere
  Konfidenzwerte** — wichtig, weil die App sie als „Konfidenz" anzeigt.

## 5. Der Datensatz — und warum genau dieser

Entscheidend war, das **Gerät der Auslieferung zu treffen**:

- **RecoFit (Microsoft):** Start, aber **Unterarm-Armband** statt Uhr → physische
  Sensor-Platzierungslücke, reale Apple-Watch-Curls falsch erkannt → verworfen.
- **MM-Fit:** Handgelenk-Smartwatch, aber **Wear-OS TicWatch** → Geräte-/
  Orientierungslücke zur Apple Watch → abgelöst.
- **Kaggle „Gym Workout IMU" (aktuell):** **Apple Watch SE, 100 Hz, Handgelenk** —
  dieselbe Gerätefamilie wie im Einsatz. Das ist der Grund, warum echte Uploads
  funktionieren.

**Zentrale Einschränkung:** Der Anker ist **eine einzige Person** (single-subject).

## 6. Wieso 3 möglichst unterschiedliche Übungen?

`bicep_curl`, `tricep_extension`, `row` decken **drei klar getrennte
Bewegungsachsen** ab (Ellbogenbeugung / Überkopf-Streckung / horizontaler Zug). Bei
nur einer Trainingsperson senkt **jede zusätzliche Klasse die Trennbarkeit** —
ähnliche Übungen (Seitheben, Schulterdrücken) überlappen die Signale. Drei stark
unterschiedliche Übungen geben die saubersten Entscheidungsgrenzen und sind von
jedem Tester im Studio reproduzierbar.

## 7. Wenig Daten → Augmentation: was ist das und wofür?

Da nur **eine** Person aufgenommen hat, fehlt **Vielfalt** (andere Hand, andere
Uhr-Position, anderes Tempo). **Augmentation** erzeugt aus jedem Trainingsfenster 5
leicht veränderte Kopien (→ 6×):

- **3-D-Rotation** → andere Uhr-Orientierung / Händigkeit
- **Time-Warp** → anderes Wiederholungstempo
- **Achsen-Spiegelung** → anderes Handgelenk
- **Jitter** → Sensor-/Körperrauschen

So **simulieren wir die Streuung, die ein Mehr-Personen-Datensatz hätte**. Wichtig:
Nur Trainingsfenster werden augmentiert; Testfenster bleiben unberührt und ihre
Kopien sind aus dem Training ausgeschlossen → die Bewertung bleibt leckfrei.

## 8. Wieso ein Energie-Gate für Pausen (statt einer gelernten Klasse)?

Pausen werden **nicht antrainiert**, sondern über einen **Energie-Schwellwert**
erkannt (wenig Bewegung = Pause: Accel-Magnitude-Std > 0,08 g *oder*
Gyro-Magnitude-Mittel > 0,30 rad/s). Ein früher gelerntes `rest` hat auf echten
Uploads massiv über-ausgelöst, weil das Ruheverhalten neuer Personen (zappeln, Uhr
richten, trinken) ganz anders aussieht als die Trainings-Ruhe. Der Schwellwert ist
**geräte- und personenunabhängig** und überträgt sich daher viel besser. Zwei
weitere Sicherungen:

- **`uncertain`** bei Konfidenz < 0,50 (ehrliche Enthaltung).
- **`unknown`** via Novelty-Detektor für Bewegungen, die zu keiner bekannten Klasse
  passen (z. B. Kniebeugen) — statt sie in eine der 3 Klassen zu pressen.

## 9. Wie messen wir die Güte — und wie gut ist es?

**Leave-one-set-out-Kreuzvalidierung:** Jede Aufnahmedatei ist ein „Satz"; das
Modell trainiert auf allen anderen und testet **nur** auf dem ausgelassenen Satz.
Das ist die **ehrlichste leckfreie** Schätzung bei einer Person — **macro-F1 ≈
0,78**. Zufällige Fenster-Splits würden Werte künstlich aufblähen (Fenster derselben
Aufnahme gleichzeitig in Train und Test) → verworfen.

## 10. Limitationen (ehrlich)

- **Single-subject:** Auf einer **neuen Person** ist die reale Genauigkeit
  **niedriger** als 0,78. Echtes „leave-one-subject-out" ist unmöglich (nur eine
  Person, eine Uhr).
- **Nur 3 Übungen** (Closed-Set); alles andere soll als `unknown`/`uncertain`
  herauskommen, nicht falsch als eine der 3.
- **Gegenmittel:** invariante Features + Augmentation; der robuste Fix sind **eigene
  gelabelte Aufnahmen** (→ Continual Learning).

## 11. Continual Learning (in einem Satz)

Eigene Apple-Watch-Aufnahmen werden als **Dateien** (nicht Modelle) committet
(`data/Testdaten/<Übung>/`, Ordnername = Label). `make update` baut das Modell
**deterministisch neu** über dieselbe Pipeline → das ganze Team konvergiert auf
**ein** Modell statt „jeder Laptop hat ein eigenes". Kein Live-Nachtrainieren während
einer Demo.

## 12. Unsere wichtigsten Learnings

1. **Gerät schlägt Datenmenge.** Erst der Wechsel auf einen **Apple-Watch**-Anker
   ließ echte Uploads funktionieren — Sensor-Platzierung/Gerät sind wichtiger als ein
   größerer, aber fremder Datensatz.
2. **Robustheit per Regel statt per Klasse.** Pausen über einen Energie-Schwellwert
   zu erkennen überträgt sich besser über Personen als eine gelernte `rest`-Klasse.
3. **Wenige, klar getrennte Klassen** sind bei dünner Datenlage besser als viele
   ähnliche.
4. **Augmentation ersetzt teilweise fehlende Personen-Vielfalt** — als Ergänzung,
   nicht als vollwertiger Ersatz für echte zweite Personen.
5. **Ehrlichkeit > schöne Zahlen.** Leckfreie Bewertung, sichtbar gemachte
   Limitationen und `uncertain`/`unknown` statt erzwungener Antworten machen das
   System vertrauenswürdig.
6. **Ein Modell, geteilte Pipeline.** Identischer Code in Training und App + Modell
   als reproduzierbares Build-Artefakt verhindern Divergenz und „läuft nur bei mir".

## 13. Wo finde ich was?

| Thema | Datei |
|------|------|
| Schnellstart / Überblick | `README.md` |
| Alle Entscheidungen + Begründungen | `docs/DECISIONS.md` |
| Architektur & Datenfluss | `docs/architecture/architecture.md` |
| CRISP-DM-Verlauf | `docs/project/crisp_dm_log.md` |
| Feature-/Daten-Lexikon | `docs/data/data_dictionary.md` |
| Eigene Daten aufnehmen | `docs/project/apple_watch_data_collection_guide.md` |
| Ordner-Übersicht des Repos | `STRUCTURE.md` |
| Pipeline-Code (Training = App) | `src/ml4b/` |
| Web-App | `app/streamlit_app.py` |
