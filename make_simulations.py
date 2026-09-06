"""Erzeugt die Simulations-Videos der Doku.

    python docs/make_simulations.py

Erzeugt in docs/simulationen/:
    gesicht_simulation.mp4     Gesichtserkennung gegen faces.json (Elon Musk, InsightFace)
    fahndung_simulation.mp4    Live-Match gegen Bundespolizei-DB (InsightFace)
    web_app_simulation.mp4     Web-App: echte Seiten-Screenshots (Playwright)
    dieb_simulation.mp4        Dieb-Erkennung: echter Pose-Detektor (COCO-17-Keypoints
                               + Loitering + Gesten) auf der vorhandenen
                               Aufnahme dieb_erkennung/output/vid_dieb_guard.mp4

Alle Videos: 1280x720 (Dieb: 1280-proportional), 20 fps. Gesicht + Fahndung
nutzen ECHTE Embeddings, die direkt hier berechnet werden (kein Fake-Score).
Die Dieb-Simulation laesst den ECHTEN Code aus `dieb_erkennung/pose_detector.py`
ueber die bestehende Kameraaufnahme laufen (kein neues Video noetig).
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

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


def _to_h264(path: str) -> None:
    """mp4v (MPEG-4 Part 2) kann ein Browser nicht abspielen - er braucht
    H.264 (avc1). Re-encode daher nach dem Schreiben per gebuendeltem ffmpeg
    (imageio-ffmpeg), falls verfuegbar."""
    try:
        import imageio_ffmpeg
    except Exception:                    # pragma: no cover
        print(f"WARN: imageio-ffmpeg fehlt, {path} bleibt mp4v")
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, os.path.basename(path) + ".h264.mp4")
        cmd = [ffmpeg, "-y", "-i", path, "-c:v", "libx264", "-preset", "medium",
               "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-an", tmp]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:            # pragma: no cover
            print(f"WARN: ffmpeg-Reencode fehlgeschlagen fuer {path}")
            print(r.stderr[-800:])
            return
        shutil.move(tmp, path)
    print("  (re-encoded -> H.264/avc1)")


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


def _load_face_db(path=None):
    """Liest data/face_db/faces.json -> Liste von {embedding, title, image}."""
    path = path if path else os.path.join(ROOT, "data", "face_db", "faces.json")
    if not os.path.exists(path):
        return []
    raw = json.load(open(path, encoding="utf-8"))
    entries = []
    for name, items in raw.items():
        for item in items:
            emb = np.asarray(item.get("embedding", []), dtype=np.float32)
            if emb.size == 512:
                entries.append({
                    "embedding": emb,
                    "title": name,
                    "beschreibung": item.get("image", ""),
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


def _cover_box(img, x0, y0, w, h, title, fill=(255, 255, 255),
               border=(213, 223, 231), title_col=(36, 59, 83)):
    """Minimalistischer Karte-Kasten mit Titel ueber der Box."""
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), fill, -1)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), border, 2)
    cv2.putText(img, title, (x0 + 16, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, title_col, 1, cv2.LINE_AA)


def _draw_overlay(img, entries, scores, highlight):
    """Minimalistische Match-Liste mit Score-Balken (kein Flicker)."""
    cv2.putText(img, "Vergleich gegen faces.json", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
    x, y0 = 30, 80
    for idx, (entry, score) in enumerate(zip(entries, scores)):
        bar_x, bar_max = x + 260, 620
        fill = int(round(score * bar_max))
        col = (108, 123, 217) if score >= 0.40 else (160, 175, 187)
        cv2.rectangle(img, (x, y0 + idx * 36), (x + bar_max, y0 + idx * 36 + 22), (213, 223, 231), 1)
        cv2.rectangle(img, (x, y0 + idx * 36), (x + fill, y0 + idx * 36 + 22), col, -1)
        cv2.putText(img, f"{score:.2f}", (x + bar_max + 12, y0 + idx * 36 + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 59, 83), 1, cv2.LINE_AA)
        name = entry["title"][:34]
        cv2.putText(img, name, (x, y0 + idx * 36 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (92, 112, 128), 1, cv2.LINE_AA)


def make_gesicht_simulation(out_path):
    if FaceAnalysis is None:
        raise SystemExit("insightface fehlt")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cam_path = os.path.join(ROOT, "data", "face_db", "photos",
                            "Elon_Musk_camera.jpg")
    img = cv2.imread(cam_path)
    if img is None:
        raise SystemExit(f"Kamera-Foto fehlt: {cam_path}")
    faces = app.get(img)
    if not faces:
        raise SystemExit("kein Gesicht im Kamera-Foto")
    img_h, img_w = img.shape[:2]
    face = faces[0]
    emb = np.asarray(face.embedding, np.float32)
    emb /= np.linalg.norm(emb)

    db = _load_face_db()
    if not db:
        raise SystemExit("keine Gesichtsdatenbank (data/face_db/faces.json)")
    sample = sorted(db, key=lambda e: _cos(emb, e["embedding"]), reverse=True)[:6]
    scores = [_cos(emb, e["embedding"]) for e in sample]

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit(f"VideoWriter konnte {out_path} nicht oeffnen")

    bg = np.full((H, W, 3), _bg, np.uint8)
    l, t, r, b = _face_box(face, img_w, img_h)

    for i in range(_sec(0.5)):
        writer.write(bg)

    # --- Phase 1: Foto mit erkanntem Gesicht (statisch, minimal) ---
    overlay = img.copy()
    cv2.rectangle(overlay, (l, t), (r, b), (90, 184, 138), 2)
    cv2.putText(overlay, "Gesicht erkannt", (l, max(20, t - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 184, 138), 2, cv2.LINE_AA)

    hh, ww = overlay.shape[:2]
    ratio = min(560 / ww, 340 / hh)
    small = cv2.resize(overlay, (int(ww * ratio), int(hh * ratio)))
    x0, y0 = 40, (H - small.shape[0]) // 2 - 20

    for i in range(_sec(4.0)):
        widget = bg.copy()
        cv2.putText(widget, "GESICHTSERKENNUNG", (30, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, "Kamera-Person", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 59, 83), 1, cv2.LINE_AA)
        cv2.rectangle(widget, (x0, y0),
                      (x0 + small.shape[1], y0 + small.shape[0]),
                      (213, 223, 231), 2)
        widget[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
        cv2.putText(widget, "Embedding 512-dim  ->  Cosine-Vergleich",
                    (x0, y0 + small.shape[0] + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (92, 112, 128), 1, cv2.LINE_AA)
        writer.write(widget)

    # --- Phase 2: Vergleich gegen Gesichtsdatenbank (minimal) ---
    for i in range(_sec(4.5)):
        widget = bg.copy()
        cv2.putText(widget, "Vergleich gegen faces.json", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 59, 83), 2, cv2.LINE_AA)
        prog = min(1.0, i / _sec(2.6))
        k = int(prog * len(scores))
        _draw_overlay(widget, sample, [s if idx < k else 0 for idx, s in enumerate(scores)],
                      0 if prog >= 1 else -1)
        writer.write(widget)

    # --- Phase 3: Ergebnis (minimale Karte) ---
    best = sample[0]
    best_score = scores[0]
    label = best["title"][:40] if best_score >= 0.40 else "Unbekannt"
    col = (108, 123, 217) if best_score >= 0.40 else (36, 59, 83)
    cx0, cy0, cw, ch = W // 2 - 260, 210, 520, 180
    for i in range(_sec(3.0)):
        widget = bg.copy()
        cv2.rectangle(widget, (cx0, cy0), (cx0 + cw, cy0 + ch), (255, 255, 255), -1)
        cv2.rectangle(widget, (cx0, cy0), (cx0 + cw, cy0 + ch), (213, 223, 231), 2)
        cv2.putText(widget, "ERGEBNIS", (cx0 + 24, cy0 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (92, 112, 128), 1, cv2.LINE_AA)
        cv2.putText(widget, label, (cx0 + 24, cy0 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)
        cv2.putText(widget, f"Score: {best_score:.2f}", (cx0 + 24, cy0 + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
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
        cv2.putText(widget, "FAHNDUNG", (30, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
        hh, ww = img.shape[:2]
        ratio = min(400 / ww, 340 / hh)
        small = cv2.resize(img, (int(ww * ratio), int(hh * ratio)))
        x0, y0 = 40, 120
        cv2.putText(widget, "Kamera-Person", (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 59, 83), 1, cv2.LINE_AA)
        cv2.rectangle(widget, (x0, y0), (x0 + small.shape[1], y0 + small.shape[0]),
                      (90, 184, 138), 2)
        widget[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small

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
    cx0, cy0, cw, ch = W // 2 - 320, 180, 640, 240
    for i in range(_sec(4.5)):
        widget = bg.copy()
        cv2.rectangle(widget, (cx0, cy0), (cx0 + cw, cy0 + ch), (255, 255, 255), -1)
        cv2.rectangle(widget, (cx0, cy0), (cx0 + cw, cy0 + ch), (217, 123, 108), 2)
        cv2.putText(widget, "FAHNDUNGS-TREFFER", (cx0 + 24, cy0 + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (217, 123, 108), 2, cv2.LINE_AA)
        cv2.putText(widget, best["title"][:60], (cx0 + 24, cy0 + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, f"Cosine-Score: {best_score:.2f}  (>= 0.35)",
                    (cx0 + 24, cy0 + 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 59, 83), 2, cv2.LINE_AA)
        cv2.putText(widget, "Screenshot gespeichert  |  Telegram-Alarm versendet",
                    (cx0 + 24, cy0 + 224),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (92, 112, 128), 1, cv2.LINE_AA)
        writer.write(widget)

    for i in range(_sec(0.8)):
        writer.write(bg)
    writer.release()
    print("fahndung:", out_path, os.path.getsize(out_path) // 1024, "KB")


def make_dieb_simulation(out_path, source_video=None):
    """Neue Dieb-Simulation MIT echtem Code: Der echte Pose-Detektor
    (ShopliftingPoseDetector aus dieb_erkennung/pose_detector.py) verarbeitet
    die vorhandene Kameraaufnahme und zeichnet dabei KEIN Fake-Mockup, sondern
    die echten COCO-17-Keypoints, Loitering-/Gesten-Bewertung, Bounding-Boxen
    und Verdachts-Banner.

    source_video: vorhandene Aufnahme (Standard: dieb_erkennung/output/
    vid_dieb_guard.mp4). Es wird KEIN neues Video aufgenommen.
    """
    import sys
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from dieb_erkennung.pose_detector import ShopliftingPoseDetector

    src = source_video or os.path.join(ROOT, "dieb_erkennung", "output",
                                       "vid_dieb_guard.mp4")
    if not os.path.exists(src):
        raise SystemExit(f"Dieb-Quellvideo fehlt: {src}")

    det = ShopliftingPoseDetector(conf=0.25, loitering_threshold=3.0,
                                  gesture_frames_threshold=8, face_check=False)
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 1280.0 / sw if sw else 1.0
    ow, oh = sw, sh
    if scale > 1.0:
        ow, oh = 1280, int(round(sh * scale))
    cap.release()

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (ow, oh))
    state = {"start": {}, "history": defaultdict(list), "sus": {},
             "notified": {}, "gesture": {}, "kps": {}}
    cap = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    alert_log = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        t = frame_count / fps
        processed, sus, alerts = det.process_frame(frame, state, t)
        if scale != 1.0:
            processed = cv2.resize(processed, (ow, oh),
                                   interpolation=cv2.INTER_LINEAR)
        for a in alerts:
            msg = f"Frame {frame_count}: {a['message']}"
            print("Dieb-Alarm", msg)
            alert_log.append(msg)
        writer.write(processed)
        if frame_count % max(1, int(total / 20)) == 0:
            print(f"  dieb: {frame_count}/{total}")
    cap.release()
    writer.release()
    result = {
        "quelle": os.path.basename(src),
        "frames": frame_count,
        "fps": fps,
        "dauer_sek": frame_count / fps,
        "aufloesung": f"{sw}x{sh}",
        "conf": det.conf,
        "loitering_threshold": det.loitering_threshold,
        "gesture_frames_threshold": det.gesture_frames_threshold,
        "verdaechtige": sorted(state["sus"].keys()),
        "alerts": alert_log,
    }
    dbg = os.path.join(SIMDIR, "dieb_simulation_result.json")
    json.dump(result, open(dbg, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print("dieb:", out_path, os.path.getsize(out_path) // 1024, "KB")
    print("      PROTOKOLL:", dbg)


def _web_shots(out_dir, pages=None, width=1440, height=900, port=5199):
    """Startet die echte Web-App (Flask) und macht Vollbild-Screenshots."""
    import urllib.request
    import subprocess

    pages = pages or ["/", "/auto_track", "/waffen", "/dieb", "/gesicht",
                      "/Fahndung", "/faces", "/einstellungen"]
    server = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "web_app", "server.py"),
         "--no-webcam", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit(f"Web-App konnte auf Port {port} nicht starten")

        from playwright.sync_api import sync_playwright
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        if not os.path.exists(chrome):
            raise SystemExit("Chrome nicht gefunden (fuer Web-App-Screenshots)")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=chrome, headless=True,
                args=["--no-sandbox", "--disable-gpu"])
            page = browser.new_page(viewport={"width": width, "height": height})
            shot_paths = []
            for p in pages:
                try:
                    page.goto(f"http://127.0.0.1:{port}{p}",
                              wait_until="domcontentloaded", timeout=20000)
                    page.wait_for_timeout(1500)
                except Exception as e:
                    print("WARN: Seite nicht geladen:", p, str(e)[:100])
                    continue
                name = ("index" if p == "/" else p.strip("/").replace("/", "_"))
                out = os.path.join(out_dir, name + ".png")
                page.screenshot(path=out, full_page=True)
                shot_paths.append(out)
                print("  webapp shot:", name)
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except Exception:
            server.kill()
    return shot_paths


def _pan_frames(img, dur, ease_sec=1.0):
    """Sanftes Hoch-/Runterscrollen durch ein hohes Seiten-Screenshot.

    img kann hoeher als H sein (FullPage-Screenshot); wir fahren kontinuierlich
    von oben nach unten und wechseln nach `dur` Sekunden zur naechsten Seite.
    """
    h, w = img.shape[:2]
    scale = max(W / w, H / h)
    nw, nh = int(w * scale), int(h * scale)
    big = cv2.resize(img, (nw, nh))
    x0 = (nw - W) // 2
    y_max = max(0, nh - H)
    n = int(dur * FPS)
    frames = []
    hold = max(4, int(ease_sec * FPS))
    for i in range(n + hold):
        prog = min(1.0, i / max(1, n))
        ease = prog if y_max == 0 else prog * prog * (3 - 2 * prog)
        y0 = int(ease * y_max)
        frames.append(big[y0:y0 + H, x0:x0 + W].copy())
    return frames


def _crossfade(a, b, n):
    out = []
    for i in range(n):
        t = (i + 1) / (n + 1)
        out.append(cv2.addWeighted(a, 1 - t, b, t, 0))
    return out


def make_web_app_simulation(out_path):
    """Web-App-Simulation aus ECHTEN Screenshots der laufenden App.

    Startet serverseitig die Flask-App und photographiert jede Seite mit
    Playwright (Chrome). Das Video scrollt sanft durch jede Seite und blendet
    per Crossfade zur Naechsten ueber.
    """
    with tempfile.TemporaryDirectory() as td:
        shots = _web_shots(td)
        if not shots:
            raise SystemExit("keine Web-App-Screenshots erstellt")

        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (W, H))
        if not writer.isOpened():
            raise SystemExit(f"VideoWriter konnte {out_path} nicht oeffnen")

        first = True
        prev = None
        dur_each = 3.4
        for idx, shot in enumerate(shots):
            img = cv2.imread(shot)
            if img is None:
                continue
            frames = _pan_frames(img, dur_each)
            if prev is not None:
                for f in _crossfade(prev, frames[0], _sec(0.6)):
                    writer.write(f)
            for f in frames:
                writer.write(f)
            prev = frames[-1]
            print(f"  webapp: {os.path.basename(shot)} "
                  f"({len(frames)} Frames)")
        if prev is not None:
            for i in range(_sec(0.8)):
                writer.write(prev)
        writer.release()
    print("web_app:", out_path, os.path.getsize(out_path) // 1024, "KB")


def main() -> int:
    os.makedirs(SIMDIR, exist_ok=True)
    gesicht_path = os.path.join(SIMDIR, "gesicht_simulation.mp4")
    fahndung_path = os.path.join(SIMDIR, "fahndung_simulation.mp4")
    webapp_path = os.path.join(SIMDIR, "web_app_simulation.mp4")
    dieb_path = os.path.join(SIMDIR, "dieb_simulation.mp4")

    make_gesicht_simulation(gesicht_path)
    _to_h264(gesicht_path)
    make_fahndung_simulation(fahndung_path)
    _to_h264(fahndung_path)
    make_web_app_simulation(webapp_path)
    _to_h264(webapp_path)

    if os.path.exists(dieb_path):
        os.remove(dieb_path)
    make_dieb_simulation(dieb_path)
    _to_h264(dieb_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())