# :material-rocket-launch: Schnellstart

Alle Beispiele laufen vom **Projektstamm**:

```powershell
cd C:\Users\youss\Desktop\UtutCam
```

!!! ututcam "Das komplette System"
    Die Web-App startet alle Worker (Dieb, Waffen, Gesicht, Fahndung, Track)
    in eigenen Threads und zeigt sie im Browser an:

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe web_app\server.py
    ```

    Browser: **http://localhost:5000**

## 2) :material-webcam: Live auf der Webcam (alle Funktionen)

Mit einem Befehl startest du **Pistolen-, Messer-, Gesichts-, Fahndungs- und
Dieb-Erkennung (mit COCO-17-Keypoints)** gleichzeitig auf der Webcam
(4 Fenster):

```powershell
cd C:\Users\youss\Desktop\UtutCam
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe messer_erkennung\live_all.py
```

Fenster: `Pistolen-Erkennung` · `Messer-Erkennung` · `Gesicht + Fahndung` ·
`Dieb-Erkennung (Keypoints)`. `q` beendet alle Fenster.

> Die Dieb-Erkennung läuft dabei **immer** im 4. Fenster — Loitering +
> Gestenbewertung (Keypoints), lila Kasten + rotes Banner bei Verdacht.

## 3) :material-factory: Nur eine Funktion testen

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

=== "Pistole + Messer auf der Webcam"

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe messer_erkennung\live_3fenster.py
    ```

    3 Fenster: Waffe (Pistole) · Messer · Gesicht + Fahndung. `q` beendet.

=== "Gesichtserkennung auf der Webcam"

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe apps\face_recognition_app.py
    ```

    1 Fenster: `Face Recognition` — nur Gesichter (Namen aus `faces.json`,
    grün = Treffer, rot = Unbekannt). `q` beendet, `a` speichert die Person.

=== "Dieberkennung auf der Webcam"

    ```powershell
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe dieb_erkennung\detector.py --webcam     # Loitering (grün → lila)
    C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe dieb_erkennung\pose_detector.py --webcam  # Loitering + Keypoints + Geste
    ```

    `q` beendet das Dieb-Fenster.

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

## 4) :material-view-dashboard: Dashboard (Hauptprogramm)

```powershell
C:\Users\youss\AppData\Local\Python\pythoncore-3.14-64\python.exe main.py
```

---

Siehe die jeweilige [Funktionsseite](auto_track.md) für Details, Startbefehle,
Konfiguration und Modelle.