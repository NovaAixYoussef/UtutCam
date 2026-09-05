# Schnellstart

Alle Beispiele laufen vom **Projektstamm**:

```powershell
cd C:\Users\youss\Desktop\UtutCam
```

## 1) Komplettes System (empfohlen)

Die Web-App startet alle Worker (Dieb, Waffen, Gesicht, Fahndung, Track)
in eigenen Threads und zeigt sie im Browser an.

```powershell
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe web_app\server.py
```

Browser: **http://localhost:5000**

## 2) Nur eine Funktion testen

=== "Auto Track (echte Kamera)"

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe auto_track\auto_scan_track_cgi.py
    ```

    Tasten: `S` Scan · `T` Track · `Q`/`Esc` Beenden

=== "Auto Track (Simulation)"

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe auto_track\sim_ptz_test.py
    ```

    Zeichnet die Track-Schleife in einer simulierten Welt und nimmt sich als
    Video auf (`auto_track/sim_aufnahme.mp4`, ~50 s).

=== "Gesichtserkennung"

    ```python
    from face_erkennung.face_database import FaceDatabase
    db = FaceDatabase()
    name = db.identify(embedding)     # "Anna" oder None
    ```

=== "Waffenerkennung"

    ```python
    from waffen_erkennung.detector import GunDetector
    treffer = GunDetector().detect(frame)
    ```

=== "Dieberkennung"

    ```python
    from dieb_erkennung.detector import ShopliftingDetector
    det = ShopliftingDetector()
    verdacht = det.detect_all(frame)
    ```

=== "Fahndung (Daten aktualisieren)"

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe -c `
      "from fahndung import fahndung_downloader as f; f.refresh_fahndungen()"
    ```

## 3) Dashboard (Hauptprogramm)

```powershell
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe main.py
```

---

Siehe die jeweilige [Funktionsseite](auto_track.md) für Details, Startbefehle,
Konfiguration und Modelle.