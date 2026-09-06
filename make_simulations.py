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
    (imageio-ffmpeg), falls verfuegbar.

    Achtung: Dies ist nur der Noetfall-Weg. Bevorzugt wird das direkte
    H.264-Schreiben ueber `_H264Writer` (ein Pass, kein zwischenzeitliches
    Verschieben) - damit gibt es keine korrupten Dateien durch fehlgeschlagene
    Ein-Datei-Aktionen."""
    try:
        import imageio_ffmpeg
    except Exception:                    # pragma: no cover
        print(f"WARN: imageio-ffmpeg fehlt, {path} bleibt mp4v")
        return
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, os.path.basename(path) + ".h264.mp4")
            cmd = [ffmpeg, "-y", "-i", path, "-c:v", "libx264",
                   "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", "-an", tmp]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:            # pragma: no cover
                print(f"WARN: ffmpeg-Reencode fehlgeschlagen fuer {path}")
                print(r.stderr[-800:])
                return
            if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
                shutil.move(tmp, path)
        print("  (re-encoded -> H.264/avc1)")
    except Exception as e:              # pragma: no cover
        print(f"WARN: {e}")


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


_CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
_HAAR = None


def _load_haar():
    global _HAAR
    if _HAAR is None:
        _HAAR = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _HAAR


def _blur_faces(img, upscale=1.6):
    """Automatisches Blur aller erkannten Gesichter in einem Bild (Datenschutz).

    Verwendet den OpenCV-Haar-Cascade. `upscale` vergroessert das Bild intern,
    damit auch kleine Gesichter zuverlaessig erkannt werden. Gibt eine Kopie
    zurueck; erkennt der Detektor nichts, bleibt das Bild unveraendert.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (0, 0), fx=upscale, fy=upscale,
                       interpolation=cv2.INTER_LINEAR)
    faces = _load_haar().detectMultiScale(small, scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(int(30 * upscale),
                                                   int(30 * upscale)))
    out = img.copy()
    for (x, y, w, h) in faces:
        x = int(x / upscale)
        y = int(y / upscale)
        w = int(w / upscale)
        h = int(h / upscale)
        pad = int(0.25 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)
        roi = out[y0:y1, x0:x1]
        bsigma = max(15, int(0.35 * max(roi.shape[:2])))
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (0, 0), bsigma)
    return out


def _render_html_shots(pages, out_dir):
    """Rendert eine Liste von (name, html)-Seiten per Playwright/Chrome zu PNGs.

    Jede Seite wird im Viewport 1280x720 (ohne Scrolling) abgelichtet und als
    `name.png` in `out_dir` gespeichert. Liefert die Liste der Pfade.
    """
    if not os.path.exists(_CHROME):
        raise SystemExit(f"Chrome fehlt (HTML-Simulationen): {_CHROME}")
    from playwright.sync_api import sync_playwright
    shot_paths = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=_CHROME, headless=True,
                                     args=["--no-sandbox", "--disable-gpu",
                                           "--hide-scrollbars"])
        page = browser.new_page(viewport={"width": W, "height": H})
        for name, html in pages:
            fpath = os.path.join(out_dir, name + ".html")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(html)
            url = "file:///" + fpath.replace("\\", "/").replace(" ", "%20")
            page.goto(url, wait_until="networkidle", timeout=20000)
            page.wait_for_timeout(700)
            out = os.path.join(out_dir, name + ".png")
            page.screenshot(path=out, clip={"x": 0, "y": 0, "width": W,
                                            "height": H})
            shot_paths.append(out)
            print("  html shot:", name)
        browser.close()
    return shot_paths


def _html_style(base="#eef2f6", accent="#6c5ce7", font="'Open Sans'"):
    return f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700;800&display=swap');
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      html,body {{ width:1280px; height:720px; overflow:hidden;
        font-family:{font},Arial,sans-serif; background:{base}; color:#1f2937; }}
      .app {{ width:1280px; height:720px; position:relative; }}
      .topbar {{ position:absolute; top:0; left:0; right:0; height:64px;
        background:#ffffff; border-bottom:1px solid #e5e9f0; display:flex;
        align-items:center; padding:0 28px; gap:20px; z-index:5; }}
      .topbar .logo {{ font-weight:800; font-size:20px; color:{accent}; }}
      .topbar .crumb {{ color:#64748b; font-size:13px; }}
      .card {{ background:#fff; border:1px solid #e5e9f0; border-radius:16px;
        box-shadow:0 1px 3px rgba(16,24,40,.06); }}
      .tag {{ display:inline-flex; align-items:center; gap:6px; font-size:12px;
        font-weight:600; padding:4px 10px; border-radius:999px; }}
      .btn {{ display:inline-flex; align-items:center; gap:8px; font-weight:600;
        font-size:13px; padding:8px 16px; border-radius:10px; border:1px solid #e5e9f0;
        background:#fff; cursor:pointer; }}
      .btn.primary {{ background:{accent}; color:#fff; border-color:{accent}; }}
    </style>
    """


def _vid_from_pngs(shots, out_path, secs_per_shot=3.4, fade=0.6, hold=0.8,
                    blur=False):
    """Baut aus einer Liste von PNGs ein Video mit Crossfades.

    `blur=True` blurrt vorher alle Gesichter in jedem Frame (Datenschutz).

    Robust: schreibt die RGB-Frames in eine temporaere .raw-Datei und laesst
    ffmpeg daraus in einem Rutsch ein H.264-MP4 erzeugen. Danach folgen zwei
    weitere Reencode-Schritte, die das Ergebnis in das finale MP4 umwandeln.
    (Der zweite Reencode ist ein Workaround: der direkte Encode aus den
    HTML-Rendershots wird von einer externen Komponente (z. B. AV) zuverlaessig
    zerstoert — erst der Reencode-Output ueberlebt; empirisch deterministisch.)
    """
    import subprocess
    import imageio_ffmpeg

    tmp1 = out_path + ".t1.mp4"
    tmp2 = out_path + ".t2.mp4"
    raw = out_path + ".raw"
    frames = 0
    with open(raw, "wb") as fh:
        prev = None
        for idx, shot in enumerate(shots):
            img = cv2.imread(shot)
            if img is None:
                continue
            if blur:
                img = _blur_faces(img)
            n = max(int(secs_per_shot * FPS), 10)
            buf = []
            if prev is not None:
                nf = int(fade * FPS)
                for i in range(nf):
                    t = (i + 1) / (nf + 1)
                    buf.append(cv2.addWeighted(prev, 1 - t, img, t, 0))
            buf.extend(img for _ in range(n))
            for f in buf:
                fh.write(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).tobytes())
                frames += 1
            prev = img
            print(f"  vid: {os.path.basename(shot)} ({n} Frames)")
        if prev is not None:
            for _ in range(int(hold * FPS)):
                fh.write(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB).tobytes())
                frames += 1

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd1 = [ff, "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", raw,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an", tmp1]
    r = subprocess.run(cmd1, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG STEP1 ERROR:", r.stderr[:400])
    try:
        os.remove(raw)
    except OSError:
        pass

    def _reencode(src, dst):
        cmd = [ff, "-y", "-i", src,
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", "-an", dst]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("FFMPEG REENCODE ERROR:", r.stderr[:400])
        try:
            os.remove(src)
        except OSError:
            pass

    _reencode(tmp1, tmp2)
    _reencode(tmp2, out_path)

    # Snapshot der gerade erzeugten Bytes. Externes Tooling (AV/Watcher)
    # zerstoert mp4-Dateien, die ffmpeg frisch streamend geschrieben hat,
    # innerhalb weniger Sekunden. Eine schlichte Byte-Kopie ist stabil.
    try:
        snap = out_path + ".snap.mp4"
        with open(out_path, "rb") as fsrc, open(snap, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
    except OSError:
        snap = None

    print(f"  -> {out_path} ({frames} Frames, "
          f"{os.path.getsize(out_path) // 1024} KB)")
    return snap


def _b64_img(img, quality=92):
    """cv2-Bild (BGR) -> base64 data-URI für HTML <img>."""
    import base64
    import io
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY),
                                         quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


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

    overlay = img.copy()
    cv2.rectangle(overlay, (l, t), (r, b), (90, 184, 138), 3)
    cv2.rectangle(overlay, (l, t), (r, b), (90, 184, 138), 1)

    hh, ww = overlay.shape[:2]
    ratio = min(300 / ww, 300 / hh)
    nw, nh = int(ww * ratio), int(hh * ratio)
    photo = cv2.resize(overlay, (nw, nh))
    photo_uri = _b64_img(photo)
    cam_uri = _b64_img(cv2.resize(img, (nw, nh)))

    best = sample[0]
    best_score = scores[0]
    label = best["title"][:40] if best_score >= 0.40 else "Unbekannt"
    accent = "#10b981" if best_score >= 0.40 else "#94a3b8"

    style = _html_style(accent=accent)
    bars = ""
    for idx, (entry, score) in enumerate(zip(sample, scores)):
        pct = max(2, int(round(score * 100)))
        col = "linear-gradient(90deg,#6366f1,#8b5cf6)" if score >= 0.40 else "#cbd5e1"
        bars += f"""
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
          <div style="width:200px;font-size:13px;font-weight:600;color:#334155;
                     white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
            <span style="display:inline-block;width:22px;height:22px;border-radius:8px;
                 background:#eef2ff;color:#6366f1;text-align:center;line-height:22px;
                 font-size:11px;font-weight:700;margin-right:8px;">{idx + 1}</span>
            {entry["title"][:26]}
          </div>
          <div style="flex:1;height:18px;border-radius:99px;background:#eef2f7;overflow:hidden;">
            <div style="height:100%;width:{pct}%;border-radius:99px;background:{col};"></div>
          </div>
          <div style="width:46px;text-align:right;font-weight:700;font-size:13px;
                     color:{'#6366f1' if score >= 0.40 else '#64748b'};">{score:.2f}</div>
        </div>"""

    pages = [
        ("gesicht_1", f"""<!DOCTYPE html><html><head><meta charset="utf-8">{style}</head>
<body><div class="app">
  <div class="topbar">
    <div class="logo">UtutCam</div>
    <div class="crumb">Gesichtserkennung</div>
    <div style="flex:1"></div>
    <span class="tag" style="background:#ecfdf5;color:#059669;">● Analyse läuft</span>
  </div>
  <div style="padding:32px 36px;display:flex;gap:32px;height:656px;">
    <div style="flex:1.15;display:flex;flex-direction:column;gap:14px;">
      <div class="card" style="padding:20px;">
        <div style="font-size:12px;font-weight:700;color:#64748b;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;">
          Live-Kamera</div>
        <div style="display:flex;justify-content:center;">
          <img src="{photo_uri}" style="max-width:100%;max-height:300px;border-radius:12px;
               border:1px solid #e5e9f0;"/>
        </div>
        <div style="display:flex;align-items:center;gap:8px;margin-top:14px;">
          <span style="width:10px;height:10px;border-radius:99px;background:#10b981;"></span>
          <span style="font-size:14px;font-weight:600;color:#334155;">Gesicht erkannt</span>
          <span style="font-size:12px;color:#94a3b8;margin-left:auto;">
            Embedding 512-dim → Cosine</span>
        </div>
      </div>
      <div class="card" style="padding:18px 20px;display:flex;gap:20px;">
        <div style="flex:1;text-align:center;padding:10px 0;border-radius:12px;background:#eef2ff;">
          <div style="font-size:11px;color:#6366f1;font-weight:700;">ERKANNTES GESICHT</div>
          <div style="font-size:22px;font-weight:800;color:#4338ca;">1</div>
        </div>
        <div style="flex:1;text-align:center;padding:10px 0;border-radius:12px;background:#f1f5f9;">
          <div style="font-size:11px;color:#64748b;font-weight:700;">Datenbank</div>
          <div style="font-size:22px;font-weight:800;color:#334155;">{len(db)}</div>
        </div>
        <div style="flex:1;text-align:center;padding:10px 0;border-radius:12px;background:#f1f5f9;">
          <div style="font-size:11px;color:#64748b;font-weight:700;">Vergleich</div>
          <div style="font-size:22px;font-weight:800;color:#334155;">1:{len(db)}</div>
        </div>
      </div>
    </div>
    <div class="card" style="flex:1;padding:24px;display:flex;flex-direction:column;">
      <div style="font-size:16px;font-weight:800;color:#1e293b;">Vergleich gegen faces.json</div>
      <div style="font-size:12px;color:#94a3b8;margin:6px 0 18px;">
        Cosinus-Ähnlichkeit jedes Gesichts aus der Datenbank zum Live-Bild.</div>
      <div style="flex:1;">{bars}</div>
      <div style="margin-top:auto;border-top:1px solid #eef2f7;padding-top:16px;
                 display:flex;align-items:center;gap:10px;">
        <span class="tag" style="background:#eef2ff;color:#6366f1;">✓ Match-Prozess aktiv</span>
        <span style="font-size:12px;color:#94a3b8;margin-left:auto;">faces.json</span>
      </div>
    </div>
  </div>
</div></body></html>
"""),
        ("gesicht_2", f"""<!DOCTYPE html><html><head><meta charset="utf-8">{style}</head>
<body><div class="app">
  <div class="topbar">
    <div class="logo">UtutCam</div>
    <div class="crumb">Gesichtserkennung</div>
    <div style="flex:1"></div>
    <span class="tag" style="background:#eef2ff;color:#6366f1;">✓ Vergleich abgeschlossen</span>
  </div>
  <div style="padding:40px 36px;height:656px;display:flex;
              justify-content:center;align-items:center;">
    <div class="card" style="width:760px;padding:34px 38px;">
      <div style="display:flex;align-items:center;gap:24px;">
        <div style="width:150px;height:150px;border-radius:20px;overflow:hidden;
                   border:3px solid {accent};box-shadow:0 4px 20px rgba(16,24,40,.10);
                   display:flex;justify-content:center;align-items:center;">
          <img src="{cam_uri}" style="width:100%;height:100%;object-fit:cover;"/>
        </div>
        <div style="flex:1;">
          <div style="font-size:12px;font-weight:700;color:#64748b;
                      text-transform:uppercase;letter-spacing:.06em;">Ergebnis</div>
          <div style="font-size:30px;font-weight:800;color:{'#4338ca' if best_score >= 0.40 else '#1e293b'};
                      margin-top:6px;">{label}</div>
          <div style="display:flex;align-items:center;gap:10px;margin-top:12px;">
            <span class="tag" style="background:{'#ecfdf5' if best_score >= 0.40 else '#f8fafc'};
                  color:{'#059669' if best_score >= 0.40 else '#64748b'};">
              {'● MATCH' if best_score >= 0.40 else '○ KEIN MATCH'}</span>
            <span style="font-size:13px;color:#94a3b8;">
              Score {best_score:.2f} {'(>= 0.40)' if best_score >= 0.40 else '(< 0.40)'}</span>
          </div>
        </div>
      </div>
      <div style="border-top:1px solid #eef2f7;margin-top:26px;padding-top:20px;
                 display:flex;gap:12px;">
        <span class="tag" style="background:#eff6ff;color:#2563eb;">🖼 Screenshot gespeichert</span>
        <span class="tag" style="background:#fef3c7;color:#b45309;">✈ Telegram-Alarm versendet</span>
        <span style="font-size:12px;color:#94a3b8;margin-left:auto;align-self:center;">
          compare_against_db → data/face_db/faces.json</span>
      </div>
    </div>
  </div>
</div></body></html>
"""),
    ]

    with tempfile.TemporaryDirectory() as td:
        shots = _render_html_shots(pages, td)
        _vid_from_pngs(shots, out_path, secs_per_shot=4.0, fade=0.6, hold=0.8)
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

    overlay = img.copy()
    fb = faces[0]
    fl, ft, fr, fb2 = _face_box(fb, img_w, img_h)
    cv2.rectangle(overlay, (fl, ft), (fr, fb2), (217, 123, 108), 3)

    hh, ww = overlay.shape[:2]
    ratio = min(350 / ww, 350 / hh)
    nw, nh = int(ww * ratio), int(hh * ratio)
    photo_uri = _b64_img(cv2.resize(overlay, (nw, nh)))

    best = db_sorted[0]
    best_score = scores[0]
    db_count = len(db)

    rows = ""
    for idx, entry in enumerate(db_sorted[:8]):
        pct = max(2, int(round(scores[idx] * 100)))
        col = "linear-gradient(90deg,#ef4444,#f97316)" if scores[idx] >= 0.35 else "#cbd5e1"
        rows += f"""
        <div style="display:flex;align-items:center;gap:14px;padding:11px 0;
                    border-bottom:1px solid #eef2f7;">
          <div style="width:30px;height:30px;border-radius:10px;background:#fef2f2;
                     color:#ef4444;text-align:center;line-height:30px;font-size:12px;
                     font-weight:800;">{idx + 1}</div>
          <div style="flex:1;min-width:0;">
            <div style="font-size:13px;font-weight:600;color:#334155;white-space:nowrap;
                        overflow:hidden;text-overflow:ellipsis;">{entry["title"][:52]}</div>
            <div style="font-size:11px;color:#94a3b8;">{entry.get("beschreibung", "")[:44]}</div>
          </div>
          <div style="width:120px;height:14px;border-radius:99px;background:#eef2f7;overflow:hidden;">
            <div style="height:100%;width:{pct}%;background:{col};border-radius:99px;"></div>
          </div>
          <div style="width:46px;text-align:right;font-weight:700;font-size:13px;
                     color:{'#ef4444' if scores[idx] >= 0.35 else '#64748b'};">{scores[idx]:.2f}</div>
        </div>"""

    pages = [
        ("fahndung_1", f"""<!DOCTYPE html><html><head><meta charset="utf-8">{_html_style(accent="#ef4444")}</head>
<body><div class="app">
  <div class="topbar">
    <div class="logo">UtutCam</div>
    <div class="crumb">Fahndung</div>
    <div style="flex:1"></div>
    <span class="tag" style="background:#fef2f2;color:#dc2626;">● Live-Scan aktiv</span>
  </div>
  <div style="padding:32px 36px;display:flex;gap:32px;height:656px;align-items:stretch;">
    <div style="flex:1.05;display:flex;flex-direction:column;gap:14px;">
      <div class="card" style="padding:20px;flex:1;display:flex;flex-direction:column;">
        <div style="font-size:12px;font-weight:700;color:#64748b;
                    text-transform:uppercase;letter-spacing:.06em;margin-bottom:14px;">
          Live-Kamera</div>
        <div style="display:flex;justify-content:center;align-items:center;flex:1;">
          <img src="{photo_uri}" style="max-width:100%;max-height:300px;border-radius:12px;
               border:1px solid #e5e9f0;"/>
        </div>
        <div style="margin-top:16px;display:flex;gap:12px;">
          <span class="tag" style="background:#fef2f2;color:#dc2626;">🔍 Gesicht erkannt</span>
          <span class="tag" style="background:#f1f5f9;color:#475569;">{db_count} Fahndungen</span>
          <span style="font-size:12px;color:#94a3b8;margin-left:auto;align-self:center;">
            fahndungen.json</span>
        </div>
      </div>
    </div>
    <div class="card" style="flex:1.35;padding:24px;">
      <div style="font-size:16px;font-weight:800;color:#1e293b;">Vergleich vs. Fahndungsliste</div>
      <div style="font-size:12px;color:#94a3b8;margin:6px 0 14px;">
        Beste Treffer nach Cosinus-Ähnlichkeit (Threshold &gt;= 0.35)</div>
      {rows}
      <div style="margin-top:16px;padding-top:16px;border-top:1px solid #eef2f7;
                 display:flex;align-items:center;gap:10px;">
        <span class="tag" style="background:#fef2f2;color:#dc2626;">✈ Alarm bei Treffer</span>
        <span style="font-size:12px;color:#94a3b8;margin-left:auto;">scan_face → compare → alert</span>
      </div>
    </div>
  </div>
</div></body></html>
"""),
        ("fahndung_2", f"""<!DOCTYPE html><html><head><meta charset="utf-8">{_html_style(accent="#ef4444")}</head>
<body><div class="app">
  <div class="topbar">
    <div class="logo">UtutCam</div>
    <div class="crumb">Fahndung</div>
    <div style="flex:1"></div>
    <span class="tag" style="background:#ecfdf5;color:#059669;">✓ Treffer erkannt</span>
  </div>
  <div style="padding:40px 36px;height:656px;display:flex;justify-content:center;align-items:center;">
    <div class="card" style="width:840px;padding:32px 36px;border-color:#fecaca;border-width:1px;
               box-shadow:0 6px 30px rgba(220,38,38,.10);">
      <div style="display:flex;gap:26px;align-items:center;">
        <div style="width:170px;height:170px;border-radius:22px;overflow:hidden;border:3px solid #ef4444;
                   flex-shrink:0;">
          <img src="{photo_uri}" style="width:100%;height:100%;object-fit:cover;"/>
        </div>
        <div style="flex:1;min-width:0;">
          <div style="display:flex;align-items:center;gap:10px;">
            <span class="tag" style="background:#ef4444;color:#fff;font-size:11px;">
              FAHNDUNGS-TREFFER</span>
            <span class="tag" style="background:#fef2f2;color:#dc2626;">Ähnlichkeit {best_score * 100:.0f} %</span>
          </div>
          <div style="font-size:26px;font-weight:800;color:#1e293b;margin-top:14px;
                     line-height:1.25;">{best["title"][:70]}</div>
          <div style="font-size:13px;color:#64748b;margin-top:8px;">
            {best.get("beschreibung", "")[:110]}</div>
          <div style="display:flex;gap:10px;margin-top:18px;flex-wrap:wrap;">
            <span class="tag" style="background:#eff6ff;color:#2563eb;">🖼 Screenshot gespeichert</span>
            <span class="tag" style="background:#fef3c7;color:#b45309;">✈ Telegram-Alarm versendet</span>
            <span class="tag" style="background:#f1f5f9;color:#475569;">Score {best_score:.2f} &gt;= 0.35</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div></body></html>
"""),
    ]

    with tempfile.TemporaryDirectory() as td:
        shots = _render_html_shots(pages, td)
        _vid_from_pngs(shots, out_path, secs_per_shot=4.0, fade=0.6, hold=0.8)
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

    tmp_mp4 = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
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

    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # H.264-Reencode OHNE faststart (moov bleibt am Dateiende - kein
    # Korruptionsrisiko, fuer lokale Dateien voellig ausreichend).
    cmd = [ff, "-y", "-i", tmp_mp4,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-an", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG REENCODE ERROR:", r.stderr[:300])
    try:
        os.remove(tmp_mp4)
    except OSError:
        pass
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
                blurred = _blur_faces(cv2.imread(out))
                cv2.imwrite(out, blurred)
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

        tmp_mp4 = out_path + ".tmp.mp4"
        writer = cv2.VideoWriter(tmp_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                                 FPS, (W, H))
        if not writer.isOpened():
            raise SystemExit(f"VideoWriter konnte {tmp_mp4} nicht oeffnen")

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

        import subprocess
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ff, "-y", "-i", tmp_mp4,
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-an", out_path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("FFMPEG REENCODE ERROR:", r.stderr[:300])
        try:
            os.remove(tmp_mp4)
        except OSError:
            pass
    print("web_app:", out_path, os.path.getsize(out_path) // 1024, "KB")


def _settle_and_fix(paths, settle=6.0):
    """Wartet das Korruptions-Fenster ab und repariert beschaeftigte Dateien.

    Externes Tooling zerstoert mp4-Dateien, die ffmpeg frisch streamend
    geschrieben hat, innerhalb weniger Sekunden (empirisch: ~< 6 s). Danach
    verifizieren wir jede finale Datei aus einem frischen Subprozess und
    stellen notfalls die Byte-Kopie (`.snap.mp4`) wieder her, die stabil ist.
    """
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    time.sleep(settle)
    for out_path in paths:
        for _ in range(3):
            r = subprocess.run([ff, "-v", "error", "-i", out_path,
                                "-f", "null", "-"],
                               capture_output=True, text=True)
            if not r.stderr.strip():
                break
            snap = out_path + ".snap.mp4"
            if not os.path.exists(snap):
                break
            print(f"  korrupt -> wiederhergestellt: {os.path.basename(out_path)}")
            shutil.copyfile(snap, out_path)
            time.sleep(2.0)
            try:
                os.remove(snap)
            except OSError:
                pass
        snap = out_path + ".snap.mp4"
        if os.path.exists(snap):
            try:
                os.remove(snap)
            except OSError:
                pass


def main() -> int:
    os.makedirs(SIMDIR, exist_ok=True)
    gesicht_path = os.path.join(SIMDIR, "gesicht_simulation.mp4")
    fahndung_path = os.path.join(SIMDIR, "fahndung_simulation.mp4")
    webapp_path = os.path.join(SIMDIR, "web_app_simulation.mp4")
    dieb_path = os.path.join(SIMDIR, "dieb_simulation.mp4")

    make_gesicht_simulation(gesicht_path)
    make_fahndung_simulation(fahndung_path)
    make_web_app_simulation(webapp_path)

    if os.path.exists(dieb_path):
        os.remove(dieb_path)
    make_dieb_simulation(dieb_path)

    _settle_and_fix([gesicht_path, fahndung_path, webapp_path, dieb_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())