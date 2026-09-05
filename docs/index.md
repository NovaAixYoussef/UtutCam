# UtutCam

**Live-Videoanalysen & Alarmsystem** — Personenverfolgung mit PTZ-Kamera,
Gesichtserkennung, Waffenerkennung, Dieberkennung und Bundespolizei-Fahndung
in einem integrierten System.

---

## Was ist UtutCam?

UtutCam verarbeitet das **Live-Videobild einer IP-Kamera in Echtzeit** und
liefert für jede Funktion ein konkretes Ergebnis:

<div class="grid cards" markdown>

-   **[Auto Track](auto_track.md)**

    Der Detektor erkennt eine Person, die Kamera schwenkt (PTZ) und hält sie
    automatisch im Bild. Voll- oder Halbaufnahme automatisch gestartet.
    ![Ablauf](bilder/auto_track_ablauf.png)

-   **[Gesichtserkennung](gesichtserkennung.md)**

    InsightFace erzeugt 512-dim-Embeddings und gleicht sie gegen die lokale
    Gesichtsdatenbank ab (Cosine-Similarity).
    ![Netzwerk](bilder/gesichts_netzwerk.png)

-   **[Waffenerkennung](waffenerkennung.md)**

    YOLOv8-Modelle für Waffen, Hände und Messer mit Konfidenz-Ausgabe.
    ![Ablauf](bilder/waffen_ablauf.png)

-   **[Dieberkennung](dieberkennung.md)**

    Personen-Tracking über die Zeit (IOWA/Loitering-Analyse) erkennt
    verdächtiges Verhalten und feuert einen Alarm.
    ![Ablauf](bilder/dieb_ablauf.png)

-   **[Fahndung](fahndung.md)**

    Abgleich gegen die Bundespolizei-Fahndungsliste, bei Treffer
    Telegram-Benachrichtigung.
    ![Ablauf](bilder/fahndung_ablauf.png)

-   **[Web-App](web_app.md)**

    Browser-Dashboard mit Live-Stream, Modul-Schaltern, Event-Log und einer
    REST-API für alle Funktionen.
    ![Architektur](bilder/web_app_architektur.png)

</div>

## Schnellstart

```powershell
cd C:\Users\youss\Desktop\UtutCam
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe web_app\server.py
```

Danach im Browser: **http://localhost:5000**

## Die Modelle

Die Neuronalen Netze wurden teils **auf Google Colab** trainiert und liegen als
Ultralytics-`.pt` bereit:

| Modell | Inhalt |
|--------|--------|
| `UtutGun.pt` | Schusswaffen |
| `UtutHand.pt` / `UtutHand2.pt` | Hände |
| `UtutBest.pt` | Waffen + Hände + Messer |
| `UtutPerson.pt` | Personen (Dieb-/Auto-Track-Grundlage) |
| InsightFace `buffalo_l` | Gesichts-Embeddings (512-dim) |

Siehe [Installation](installation.md) und [Struktur](installation.md#projektstruktur).