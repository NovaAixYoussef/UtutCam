# Simulationen & Visualisierungen

Alle Grafiken und Videos unter `docs/` werden automatisch erzeugt
(`visualize_network.py`, `visualize_flow.py`, `make_*.py`), sodass sie
jederzeit neu generiert werden können.

## Simulationen (Videos)

| Video | Funktion | Inhalt | Länge |
|-------|----------|--------|-------|
| [Gesicht](simulationen/gesicht_simulation.mp4) | Gesichtserkennung | echter InsightFace-Match auf einem Foto | ~13 s |
| [Fahndung](simulationen/fahndung_simulation.mp4) | Fahndung | echter Cosinus-Match gegen `fahndungen.json` | ~9 s |
| [Waffen](simulationen/waffen_simulation.mp4) | Waffenerkennung | echte Gun-Erkennungsaufnahme | ~26 s |
| [Dieb](simulationen/dieb_simulation.mp4) | Dieberkennung | echte Loitering-Aufnahme | ~32 s |
| [Web-App](simulationen/web_app_simulation.mp4) | Web-App | Dashboard-Mockup mit Waffen-Alarm | ~7 s |
| [Auto Track](simulationen/auto_track_simulation.mp4) | Personenverfolgung | komplette PTZ-Track-Schleife | ~59 s |

## Visualisierungen (Diagramme)

| Diagramm | Funktion | Datei |
|----------|----------|-------|
| Netzwerk (Neuronales Netz, InsightFace) | Gesicht | `bilder/gesichts_netzwerk.png` (+ `.svg`) |
| MLP-Beispiel | – | `bilder/mlp_beispiel.png` |
| Ablauf Auto-Track | Personenverfolgung | `bilder/auto_track_ablauf.png` |
| Ablauf Waffen | Waffenerkennung | `bilder/waffen_ablauf.png` |
| Ablauf Dieb | Dieberkennung | `bilder/dieb_ablauf.png` |
| Ablauf Fahndung | Fahndung | `bilder/fahndung_ablauf.png` |
| Architektur Web-App | Web-App | `bilder/web_app_architektur.png` |
| Beispiel (Flow) | – | `bilder/flow_beispiel.png` |

## Neu erzeugen

```powershell
cd C:\Users\youss\Desktop\UtutCam\docs
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe make_diagrams.py      # Netzwerk-Diagramme
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe make_flows.py        # Ablauf-Diagramme
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe make_simulations.py  # Videos (echte InsightFace-Matches)
```

> Waffen-/Dieb-Videos werden aus vorhandenen echten Aufnahmen
> (`data/results/`, `dieb_erkennung/output/`) übernommen, die Simulationsvideos
> für Gesicht/Fahndung nutzen echte Embeddings von InsightFace.