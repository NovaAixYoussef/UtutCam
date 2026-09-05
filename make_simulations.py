"""Erzeugt die Simulations-Videos der Doku.

    python docs/make_simulations.py

Erzeugt in docs/simulationen/:
    gesicht_simulation.mp4     Gesichtserkennung auf echtem Foto (InsightFace)
    fahndung_simulation.mp4    Live-Match gegen Bundespolizei-DB (InsightFace)
    web_app_simulation.mp4     Dashboard-Mockup-Animation

Alle Videos: 1280x720, 20 fps. Gesicht + Fahndung nutzen ECHTE Embeddings,
die direkt hier berechnet werden (kein Fake-Score).
"""

from __future__ import annotations

import json
import math
import os
import random

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIMDIR = os.path.join(HERE, "simulationen")
ROOT = os.path.abspath(os.path.join(HERE, ".."))

W, H = 1280, 720
FPS = 20

try:
    from insightface.app import FaceAnalysis
except Exception:                      # pragma: no cover
    FaceAnalysis = None

_bg = (245, 248, 251)


def _sec(t: float) -> int:
    return int(round(t * FPS))


def _load_fahndung_db(path=None):
    path = path or os.path.join(ROOT, "data", "fahndung", "fahndungen.json")
    if not os.path.exists(path):
        return []
    db = json.load(open(path, encoding="utf-8"))
    entries = []
    for title, case in db.items():
        for item in case:
            emb = np.asarray(item.get("embedding", []), dtype=np.float32)
            if emb.size == 512:
                entries.append({
                    "embedding": emb,
                    "title": item.get("title", title),
                    "beschreibung": item.get("beschreibung", ""),
                    "image_url": item.get("image", ""),
                })
    return entries


def _pick_photo_with_face(app, max_tries=40):
    img_dir = os.path.join(ROOT, "data", "fahndung", "bilder")
    if not os.path.isdir(img_dir):
        return None, None
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rnd = random.Random(4)
    rnd.shuffle(files)
    for name in files[:max_tries]:
        path = os.path.join(img_dir, name)
        img = cv2.imread(path)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            return img, faces, path
    return None, None


def _cos(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    a /= np.linalg.norm(a) or 1
    b /= np.linalg.norm(b) or 1
    return float(np.dot(a, b))


def _face_box(face, img_w, img_h):
    l, t, r, b = (int(face.bbox[0]), int(face.bbox[1]),
                  int(face.bbox[2]), int(face.bbox[3]))
    return np.clip(l, 0, img_w), np.clip(t, 0, img_h), np.clip(r, 0, img_w), np.clip(b, 0, img_h)


def _draw_overlay(img, entries, scores, highlight):
    """Groesse die Eingangsboxen sinnvoll und zeichnet Score-Balken-Liste."""
    cv2.putText(img, "MATCH-ANALYSE", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
    x, y0 = 30, 80
    for idx, (entry, score) in enumerate(zip(entries, scores)):
        bar_x, bar_max = x + 260, 620
        fill = int(round(score * bar_max))
        col = (90, 184, 138) if score < 0.35 else (108, 123, 217)
        if idx == highlight:
            col = (108, 123, 217)
        cv2.rectangle(img, (x, y0 + idx * 36), (x + bar_max, y0 + idx * 36 + 22), (213, 223, 231), 1)
        cv2.rectangle(img, (x, y0 + idx * 36), (x + fill, y0 + idx * 36 + 22), col, -1)
        cv2.putText(img, f"{score:.2f}", (x + bar_max + 12, y0 + idx * 36 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 59, 83), 1, cv2.LINE_AA)
        name = entry["title"][:34]
        cv2.putText(img, name, (x, y0 + idx * 36 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 112, 128), 1, cv2.LINE_AA)


def _fill_marker(img, cx, cy, t):
    """animierter Scan-Punkt auf dem Live-Bild."""
    r = 8 + int(4 * math.sin(t * 6))
    cv2.circle(img, (cx, cy), r, (217, 123, 108), 2)


def make_gesicht_simulation(out_path):
    if FaceAnalysis is None:
        raise SystemExit("insightface fehlt")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img, faces, path = _pick_photo_with_face(app)
    if img is None:
        raise SystemExit("kein Foto mit Gesicht gefunden")
    img_h, img_w = img.shape[:2]
    face = faces[0]
    emb = np.asarray(face.embedding, np.float32)
    emb /= np.linalg.norm(emb)

    db = _load_fahndung_db()
    scores = sorted([_cos(emb, e["embedding"]) for e in db], reverse=True)[:6]
    sample = sorted(db[:6], key=lambda e: _cos(emb, e["embedding"]), reverse=True)
    scores = [_cos(emb, e["embedding"]) for e in sample]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter konnte {out_path} nicht oeffnen")

    bg = np.full((H, W, 3), _bg, np.uint8)
    l, t, r, b = _face_box(face, img_w, img_h)
    gx, gy = (l + r) // 2, (t + b) // 2

    for i in range(_sec(0.5)):
        writer.write(bg)

    # --- Phase 1: Foto mit erkanntem Gesicht ---
    for i in range(_sec(4.0)):
        scale = 1.0 + 0.06 * math.sin(i / _sec(1.0) * 2 * math.pi)
        overlay = img.copy()
        cx, cy = (l + r) // 2, (t + b) // 2
        w2, h2 = int((r - l) * 0.55 * scale), int((b - t) * 0.9 * scale)
        cv2.rectangle(overlay, (cx - w2, cy - h2), (cx + w2, cy + h2),
                      (90, 184, 138), 3)
        cv2.putText(overlay, "Gesicht erkannt", (cx - w2, cy - h2 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 184, 138), 2, cv2.LINE_AA)
        cv2.putText(overlay, "112x112x3 -> Embedding 512-dim", (28, H - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (92, 112, 128), 1, cv2.LINE_AA)

        widget = bg.copy()
        hh, ww = overlay.shape[:2]
        ratio = min(560 / ww, 340 / hh)
        small = cv2.resize(overlay, (int(ww * ratio), int(hh * ratio)))
        x0, y0 = 40, (H - small.shape[0]) // 2
        widget[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        cv2.rectangle(widget, (x0, y0),
                      (x0 + small.shape[1], y0 + small.shape[0]),
                      (213, 223, 231), 2)
        _fill_marker(widget, x0 + 60, y0 + 60, i / FPS)
        cv2.putText(widget, "Kamera-Person", (x0, y0 + small.shape[0] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 59, 83), 1, cv2.LINE_AA)
        ph = 0.6  # Platzhalter-Panel rechts
        writer.write(widget)

    # --- Phase 2: Vergleich gegen DB ---
    for i in range(_sec(4.5)):
        widget = bg.copy()
        cv2.putText(widget, "Cosine-Vergleich gegen Datenbank", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 59, 83), 2, cv2.LINE_AA)
        prog = min(1.0, i / _sec(2.6))
        k = int(prog * len(scores))
        _draw_overlay(widget, sample, [s if idx < k else 0 for idx, s in enumerate(scores)],
                      0 if prog >= 1 else -1)
        writer.write(widget)

    # --- Phase 3: Ergebnis ---
    best = sample[0]
    best_score = scores[0]
    label = best["title"][:40] if best_score >= 0.35 else "Unbekannt"
    col = (108, 123, 217) if best_score >= 0.35 else (36, 59, 83)
    for i in range(_sec(3.0)):
        widget = bg.copy()
        cv2.putText(widget, "ERGEBNIS", (W // 2 - 120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, label, (W // 2 - 260, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, 3, cv2.LINE_AA)
        cv2.putText(widget, f"Score: {best_score:.2f}", (W // 2 - 120, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv2.LINE_AA)
        writer.write(widget)

    for i in range(_sec(0.8)):
        writer.write(bg)

    writer.release()
    print("gesicht:", out_path, os.path.getsize(out_path) // 1024, "KB")


def make_fahndung_simulation(out_path):
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img, faces, path = _pick_photo_with_face(app)
    if img is None:
        raise SystemExit("kein Foto mit Gesicht gefunden")
    db = _load_fahndung_db()
    if not db:
        raise SystemExit("keine Fahndungs-DB")
    face = faces[0]
    emb = np.asarray(face.embedding, np.float32)
    emb /= np.linalg.norm(emb)
    db_sorted = sorted(db, key=lambda e: _cos(emb, e["embedding"]), reverse=True)[:8]
    scores = [_cos(emb, e["embedding"]) for e in db_sorted]

    img_h, img_w = img.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bg = np.full((H, W, 3), _bg, np.uint8)

    for i in range(_sec(0.5)):
        writer.write(bg)

    # Phase 1: Live-Kamera + Fahndungs-Liste lädt
    for i in range(_sec(3.5)):
        widget = bg.copy()
        hh, ww = img.shape[:2]
        ratio = min(400 / ww, 340 / hh)
        small = cv2.resize(img, (int(ww * ratio), int(hh * ratio)))
        x0, y0 = 40, 130
        widget[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        cv2.rectangle(widget, (x0, y0), (x0 + small.shape[1], y0 + small.shape[0]),
                      (90, 184, 138), 2)
        _fill_marker(widget, x0 + 60, y0 + 60, i / FPS)
        cv2.putText(widget, "LIVE-KAMERA", (x0, y0 - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 184, 138), 2, cv2.LINE_AA)

        cv2.putText(widget, "fahndungen.json", (W - 600, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, f"{len(db)} Fahndungen geladen", (W - 600, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (92, 112, 128), 1, cv2.LINE_AA)
        n_show = int(min(len(db), 8) * min(1, i / _sec(1.5)))
        for j in range(n_show):
            y = 120 + j * 56
            cv2.rectangle(widget, (W - 620, y), (W - 40, y + 44), (226, 232, 238), 1)
            cv2.putText(widget, db_sorted[j]["title"][:52], (W - 606, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 112, 128), 1, cv2.LINE_AA)
            cv2.putText(widget, "( wird verglichen ... )", (W - 606, y + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 175, 187), 1, cv2.LINE_AA)
        writer.write(widget)

    # Phase 2: Treffer-Meldung
    best = db_sorted[0]
    best_score = scores[0]
    for i in range(_sec(4.5)):
        widget = bg.copy()
        cv2.putText(widget, "FAHNDUNGS-TREFFER!", (W // 2 - 320, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (217, 123, 108), 3, cv2.LINE_AA)
        cv2.putText(widget, best["title"][:60], (W // 2 - 320, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, f"Cosine-Score: {best_score:.2f}  (>= 0.35)",
                    (W // 2 - 320, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, "Screenshot gespeichert  |  Telegram-Alarm versendet",
                    (W // 2 - 320, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (92, 112, 128), 1, cv2.LINE_AA)
        writer.write(widget)

    for i in range(_sec(0.8)):
        writer.write(bg)
    writer.release()
    print("fahndung:", out_path, os.path.getsize(out_path) // 1024, "KB")


def make_web_app_simulation(out_path):
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    bg = np.full((H, W, 3), _bg, np.uint8)
    workers = [("DiebWorker", "Shoplifting"), ("WaffenWorker", "Gun / Hand / Messer"),
               ("GesichtWorker", "452 Gesichter DB"), ("FahndungWorker", "124 Fahndungen"),
               ("TrackWorker", "PTZ aktiv")]
    N = len(workers)
    cols = [(108, 123, 217), (217, 123, 108), (90, 184, 138),
            (90, 184, 138), (90, 184, 138)]

    # HEADER + Seitenleiste
    for i in range(_sec(6.0)):
        frame = bg.copy()
        cv2.rectangle(frame, (0, 0), (W, 74), (36, 59, 83), -1)
        cv2.putText(frame, "UtutCam  -  Dashboard", (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "http://localhost:5000", (W - 420, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (213, 223, 231), 1, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 74), (240, H), (226, 232, 238), -1)
        pages = ["übersicht", "auto_track", "Waffen", "Dieb", "Gesicht",
                 "Fahndung", "Einstellungen"]
        for j, p in enumerate(pages):
            y = 100 + j * 62
            col = (36, 59, 83)
            if j == 2:
                col = (217, 123, 108)
            cv2.putText(frame, "• " + p, (24, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 1, cv2.LINE_AA)

        x0, y0 = 280, 100
        cv2.rectangle(frame, (x0, y0), (W - 20, y0 + 300), (255, 255, 255), -1)
        cv2.rectangle(frame, (x0, y0), (W - 20, y0 + 300), (213, 223, 231), 2)
        cv2.putText(frame, "LIVE-VIDEO  (MJPEG  /api/stream)", (x0 + 16, y0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 59, 83), 2, cv2.LINE_AA)
        n_lines = 4
        for k in range(n_lines):
            lx = x0 + 40 + (i * 9) % (W - x0 - 80)
            cv2.line(frame, (lx, y0 + 60 + k * 28), (lx + 90 + k * 25, y0 + 60 + k * 28),
                     (213, 223, 231), 2)
        _fill_marker(frame, x0 + 60, y0 + 70, i / FPS)

        # Worker-Karten
        for j, (name, desc) in enumerate(workers):
            wx = x0 + (j % 3) * 210
            wy = y0 + 340 + (j // 3) * 130
            on = i / FPS > 0.8 * (j + 1)
            col = (90, 184, 138) if on else (160, 175, 187)
            cv2.rectangle(frame, (wx, wy), (wx + 190, wy + 100), (255, 255, 255), -1)
            cv2.rectangle(frame, (wx, wy), (wx + 190, wy + 100), (213, 223, 231), 2)
            cv2.circle(frame, (wx + 24, wy + 26), 8, col, -1)
            cv2.putText(frame, name, (wx + 42, wy + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36, 59, 83), 1, cv2.LINE_AA)
            cv2.putText(frame, desc, (wx + 12, wy + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (92, 112, 128), 1, cv2.LINE_AA)
            cv2.putText(frame, "AKTIV" if on else "STOP", (wx + 140, wy + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        t = i / FPS
        if 3.2 < t < 5.0:
            cv2.rectangle(frame, (W - 470, 470), (W - 40, 520), (217, 123, 108), -1)
            cv2.putText(frame, "WARNUNG: Waffe erkannt  conf 0.83", (W - 450, 500),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

    for i in range(_sec(0.8)):
        writer.write(bg)
    writer.release()
    print("web_app:", out_path, os.path.getsize(out_path) // 1024, "KB")


def main() -> int:
    os.makedirs(SIMDIR, exist_ok=True)
    make_gesicht_simulation(os.path.join(SIMDIR, "gesicht_simulation.mp4"))
    make_fahndung_simulation(os.path.join(SIMDIR, "fahndung_simulation.mp4"))
    make_web_app_simulation(os.path.join(SIMDIR, "web_app_simulation.mp4"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())