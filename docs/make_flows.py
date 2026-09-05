"""Erzeugt alle Ablauf-/Architektur-Diagramme der Doku (Light-Mode).

Rueckruf jederzeit:

    python docs/make_flows.py

Erzeugt in docs/bilder/:
    auto_track_ablauf.png|svg   Auto-Track + PTZ
    waffen_ablauf.png|svg       Waffen-/Hand-/Messer-Erkennung
    dieb_ablauf.png|svg         Dieberkennung (Shoplifting)
    fahndung_ablauf.png|svg     Fahndungs-Scanner + Match
    web_app_architektur.png|svg Flask-Dashboard + Worker
"""

import os

from visualize_flow import visualize_flow

HERE = os.path.dirname(os.path.abspath(__file__))
BILDER = os.path.join(HERE, "bilder")


def auto_track():
    boxes = [
        {"id": 0, "label": "Kamera (RTSP)", "sub": "192.168.1.100", "color": "blau",
         "x": 0.085, "y": 0.62, "w": 0.14, "h": 0.30},
        {"id": 1, "label": "Frame lesen", "sub": "480x270\n(fuer HOG)", "color": "grau",
         "x": 0.275, "y": 0.62, "w": 0.13, "h": 0.34},
        {"id": 2, "label": "HOG-Detektor", "sub": "Person finden", "color": "lila",
         "x": 0.46, "y": 0.62, "w": 0.13, "h": 0.30},
        {"id": 3, "label": "EMA-Glaettung", "sub": "stabiler Kasten", "color": "lila",
         "x": 0.66, "y": 0.62, "w": 0.13, "h": 0.30},
        {"id": 4, "label": "Abweichung", "sub": "Box vs. Mitte", "color": "orange",
         "x": 0.85, "y": 0.62, "w": 0.13, "h": 0.28},
        {"id": 5, "label": "PID-Regler", "sub": "Pan-Richtung", "color": "orange",
         "x": 0.945, "y": 0.62, "w": 0.11, "h": 0.26},
        {"id": 6, "label": "PTZ steuern", "sub": "CGI left/right", "color": "gruen",
         "x": 0.95, "y": 0.18, "w": 0.11, "h": 0.26},
        {"id": 7, "label": "Loop", "sub": "0.6 s Pause", "color": "grau",
         "x": 0.66, "y": 0.12, "w": 0.10, "h": 0.20},
    ]
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5, "Pan-Fehler"),
              (5, 6), (6, 7), (7, 0, "naechster Frame")]
    return visualize_flow(
        boxes, arrows,
        output_path=os.path.join(BILDER, "auto_track_ablauf.png"),
        title="Auto-Track — Scan- und Tracking-Schleife",
        subtitle="Kamera -> HOG-Detektion -> EMA-Glaettung -> PID -> PTZ-CGI (Loop)",)


def waffen():
    boxes = [
        {"id": 0, "label": "Frame", "sub": "Kamera / Video", "color": "blau",
         "x": 0.09, "y": 0.62, "w": 0.13, "h": 0.30},
        {"id": 1, "label": "YOLOv8-Modell", "sub": "UtutGun / Hand2 / Best", "color": "lila",
         "x": 0.28, "y": 0.62, "w": 0.15, "h": 0.38},
        {"id": 2, "label": "Conf. >= 0.25?", "color": "orange",
         "x": 0.48, "y": 0.62, "w": 0.14, "h": 0.28},
        {"id": 3, "label": "Zielklasse?", "sub": "gun / hand / messer", "color": "orange",
         "x": 0.67, "y": 0.62, "w": 0.14, "h": 0.38},
        {"id": 4, "label": "Box + Label", "sub": "orangener Kasten\ngun 0.83", "color": "gruen",
         "x": 0.86, "y": 0.62, "w": 0.14, "h": 0.36},
        {"id": 5, "label": "Alarm", "sub": "Web-App / Telegram", "color": "rot",
         "x": 0.955, "y": 0.62, "w": 0.10, "h": 0.30},
        {"id": 6, "label": "Hand ueber Waffe?", "sub": "hand_cover()-Check", "color": "grau",
         "x": 0.28, "y": 0.12, "w": 0.16, "h": 0.26},
    ]
    arrows = [(0, 1), (1, 2), (2, 3, ">= 0.25"), (2, 6, "sonst"), (3, 4, "Klasse ja"),
              (3, 6, "nein"), (4, 5)]
    return visualize_flow(
        boxes, arrows,
        output_path=os.path.join(BILDER, "waffen_ablauf.png"),
        title="Waffenerkennung — YOLOv8-Pipeline",
        subtitle="Frame -> YOLO -> Konfidenz + Zielklasse -> Box/Label -> Alarm")


def dieb():
    boxes = [
        {"id": 0, "label": "Frame", "sub": "Kamera / Video", "color": "blau",
         "x": 0.09, "y": 0.62, "w": 0.13, "h": 0.30},
        {"id": 1, "label": "UtutPerson.pt", "sub": "YOLO conf 0.50", "color": "lila",
         "x": 0.27, "y": 0.62, "w": 0.15, "h": 0.34},
        {"id": 2, "label": "DeepSORT", "sub": "stabile Track-IDs", "color": "lila",
         "x": 0.47, "y": 0.62, "w": 0.14, "h": 0.34},
        {"id": 3, "label": "Loitering?", "sub": "> 10 s stabil", "color": "orange",
         "x": 0.66, "y": 0.62, "w": 0.14, "h": 0.34},
        {"id": 4, "label": "Gruener Kasten", "sub": "normale Person", "color": "gruen",
         "x": 0.66, "y": 0.14, "w": 0.14, "h": 0.26},
        {"id": 5, "label": "Lila Kasten", "sub": "Verdaechtig", "color": "rot",
         "x": 0.86, "y": 0.62, "w": 0.13, "h": 0.30},
        {"id": 6, "label": "Screenshot", "sub": "data/dieb/treffer/", "color": "grau",
         "x": 0.925, "y": 0.30, "w": 0.14, "h": 0.26},
        {"id": 7, "label": "Gesicht + Telegram", "sub": "Name, Foto, Alarm", "color": "rot",
         "x": 0.925, "y": 0.14, "w": 0.14, "h": 0.26},
    ]
    arrows = [(0, 1), (1, 2), (2, 3), (3, 4, "nein"), (3, 5, "ja"),
              (5, 6), (5, 7)]
    return visualize_flow(
        boxes, arrows,
        output_path=os.path.join(BILDER, "dieb_ablauf.png"),
        title="Dieberkennung — Personen-Tracking + Loitering",
        subtitle="YOLO-Person -> DeepSORT -> 10 s stabil -> lila Kasten + Screenshot + Alarm")


def fahndung():
    boxes = [
        {"id": 0, "label": "bundespolizei.de", "sub": "HTML-Liste", "color": "blau",
         "x": 0.14, "y": 0.78, "w": 0.16, "h": 0.28},
        {"id": 1, "label": "Scraper", "sub": "BeautifulSoup", "color": "grau",
         "x": 0.35, "y": 0.78, "w": 0.14, "h": 0.28},
        {"id": 2, "label": "Details + Fotos", "sub": "data/fahndung/bilder/", "color": "grau",
         "x": 0.56, "y": 0.78, "w": 0.16, "h": 0.32},
        {"id": 3, "label": "Embeddings", "sub": "InsightFace, 512-dim", "color": "gruen",
         "x": 0.78, "y": 0.78, "w": 0.17, "h": 0.34},
        {"id": 4, "label": "Live-Kamera", "sub": "Gesicht", "color": "blau",
         "x": 0.14, "y": 0.28, "w": 0.15, "h": 0.28},
        {"id": 5, "label": "Embedding", "sub": "Live-Gesicht, 512-dim", "color": "lila",
         "x": 0.35, "y": 0.28, "w": 0.16, "h": 0.32},
        {"id": 6, "label": "Cosinus-Vergleich", "sub": "gegen alle DB-Vektoren", "color": "orange",
         "x": 0.57, "y": 0.42, "w": 0.17, "h": 0.34},
        {"id": 7, "label": "Score >= 0.35?", "color": "orange",
         "x": 0.78, "y": 0.28, "w": 0.15, "h": 0.26},
        {"id": 8, "label": "Treffer!", "sub": "Telegram + Screenshot", "color": "rot",
         "x": 0.80, "y": 0.88, "w": 0.16, "h": 0.30},
    ]
    arrows = [(0, 1), (1, 2), (2, 3), (4, 5), (3, 6), (5, 6), (6, 7),
              (7, 8, ">= 0.35"), (7, 4, "kein Treffer")]
    return visualize_flow(
        boxes, arrows,
        output_path=os.path.join(BILDER, "fahndung_ablauf.png"),
        title="Fahndung — Bundespolizei-Scanner + Live-Match",
        subtitle="Scraper -> fahndungen.json -> Embeddings -> Cosinus-Match >= 0.35 -> Alarm")


def web_app():
    boxes = [
        {"id": 0, "label": "Browser", "sub": "localhost:5000", "color": "blau",
         "x": 0.11, "y": 0.5, "w": 0.15, "h": 0.30},
        {"id": 1, "label": "Flask", "sub": "web_app/server.py", "color": "lila",
         "x": 0.32, "y": 0.5, "w": 0.16, "h": 0.40},
        {"id": 2, "label": "REST-API", "sub": "/api/*", "color": "grau",
         "x": 0.55, "y": 0.80, "w": 0.15, "h": 0.26},
        {"id": 3, "label": "MJPEG-Stream", "sub": "/api/stream", "color": "grau",
         "x": 0.55, "y": 0.52, "w": 0.15, "h": 0.26},
        {"id": 4, "label": "DiebWorker", "sub": "Shoplifting", "color": "gruen",
         "x": 0.79, "y": 0.88, "w": 0.15, "h": 0.26},
        {"id": 5, "label": "WaffenWorker", "sub": "Gun/Hand/Messer", "color": "gruen",
         "x": 0.905, "y": 0.645, "w": 0.15, "h": 0.26},
        {"id": 6, "label": "GesichtWorker", "sub": "Face-Match", "color": "gruen",
         "x": 0.905, "y": 0.35, "w": 0.15, "h": 0.26},
        {"id": 7, "label": "FahndungWorker", "sub": "Bundespolizei", "color": "gruen",
         "x": 0.79, "y": 0.11, "w": 0.15, "h": 0.26},
        {"id": 8, "label": "TrackWorker", "sub": "PTZ", "color": "gruen",
         "x": 0.79, "y": 0.655, "w": 0.15, "h": 0.26},
        {"id": 9, "label": "Templates", "sub": "HTML, geslidet", "color": "grau",
         "x": 0.55, "y": 0.20, "w": 0.15, "h": 0.26},
    ]
    arrows = [(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8),
              (1, 9), (4, 5), (8, 6)]
    return visualize_flow(
        boxes, arrows,
        output_path=os.path.join(BILDER, "web_app_architektur.png"),
        title="Web-App — Flask-Dashboard mit Workern",
        subtitle="Browser -> Flask -> Worker-Threads (Dieb / Waffen / Gesicht / Fahndung / Track)")


def main() -> int:
    for name, fn in [("auto_track", auto_track),
                     ("waffen", waffen),
                     ("dieb", dieb),
                     ("fahndung", fahndung),
                     ("web_app", web_app)]:
        p = fn()
        print(f"{name}: {p} (+ SVG)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())