# Installation

## Voraussetzungen

- **Windows** (getestet auf Windows 11 / PowerShell)
- **Python 3.14** unter:
  `C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe`
- **Internetverbindung** für InsightFace-Modelle und die
  Bundespolizei-Fahndungsliste

## Abhängigkeiten

Die wichtigsten Pakete (`requirements`-Hinweis; das Projekt läuft direkt aus
dem Quellverzeichnis):

| Paket | Zweck |
|-------|-------|
| `opencv-python` | Video einlesen, Zeichnen, Codecs |
| `ultralytics` (YOLOv8) | Waffen-, Hand-, Messer-, Personen-Detektion |
| `insightface` + `onnxruntime` | Gesichts-Embeddings (buffalo_l) |
| `numpy` | Vektor-Mathematik, Cosinus-Vergleich |
| `flask` | Web-App / Dashboard |
| `requests` + `beautifulsoup4` | Bundespolizei-Scraper |
| `matplotlib` | Diagrammerzeugung in `docs/` |

```powershell
& "C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe" -m pip install `
    opencv-python ultralytics insightface onnxruntime numpy flask requests beautifulsoup4 matplotlib
```

??? hint "Erste InsightFace-Modelle werden beim ersten Lauf heruntergeladen"
    `~/.insightface/models/buffalo_l` (~325 MB). Einmalig nötig, danach offline.

## Geo-Position: Projektstamm

Alle Skripte erwarten den **Stammordner `C:\Users\youss\Desktop\UtutCam`**
(wegen `sys.path`-Einbindungen). Starte deshalb immer von dort:

```powershell
cd C:\Users\youss\Desktop\UtutCam
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe <skript>
```

## Projektstruktur

```
UtutCam/
├── auto_track/          Auto Track + PTZ + Detektoren
├── face_erkennung/      Gesichtserkennung (DB + Matching)
├── waffen_erkennung/    Waffen-/Hand-/Messer-Detektor + Modelle (.pt)
├── dieb_erkennung/      Shoplifting/Dieb-Detektor + Testvideos
├── fahndung/            Bundespolizei-Scanner + Match
├── messer_erkennung/    Live-Voll-Pipeline (Waffe+Gesicht+Fahndung)
├── web_app/             Flask-Server + Worker + Templates
├── core/                Config + Telegram-Notifier
├── relay/               LAN-Relay (Telegram-Zusatzkonfiguration)
├── colab/               Google-Colab-Notebook-Builder
├── data/                Datenbanken, Fotos, Fahndungen, Screenshots
└── docs/                diese Dokumentation (+ mkdocs-Site)
```

Die Modelle werden über das GitHub-Release `v1`
(`NovaAixYoussef/ututcam-models`) bereitgestellt.