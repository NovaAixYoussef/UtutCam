"""Automatischer Pipeline-/Ablauf-Diagramm-Generator (Light-Mode).

Zeichnet aus einer einfachen Box-/Pfeil-Definition ein klares Ablaufdiagramm
im gleichen hellen, gedeckten Stil wie visualize_network.py:

    [ Schritt 1 ]  →  [ Schritt 2 ]  →  [ Schritt 3 ]
                            │
                            ▼
                    [ Ergebnis / Alarm ]

Design: heller Hintergrund, gedeckte Farben (kein Neon / kein Dark-Mode),
runde Boxen, Pfeile mit Labels, kurze Texte.

Kernfunktion::

    visualize_flow(
        title    = "Mein Ablauf",
        subtitle = "Kurz beschrieben",
        boxes    = [ {"id": 0, "label": "Start",  "sub": "...", "x": .05, "y": .35, "w": .2,  "h": .3, "color": "blau"},
                     ... ],
        arrows   = [ (0, 1), (1, 2, "ja"), (1, 3, "nein"), ... ],
        output_path = "pipeline.svg",
    )
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Farben (hell, gedeckt) - passend zu visualize_network.py
# ---------------------------------------------------------------------------
PALETTE = {
    "blau":    ("#5aa7de", "#d6e9f7"),
    "lila":    ("#9d87c6", "#e7def2"),
    "gruen":   ("#5bb88a", "#dcf0e6"),
    "orange":  ("#e3a13f", "#f8ecd6"),
    "rot":     ("#d97b6c", "#f7e0db"),
    "grau":    ("#7e97ad", "#e3ebf1"),
}


def visualize_flow(
    boxes: Sequence[dict],
    arrows: Sequence[tuple],
    output_path: str = "pipeline",
    title: str = "",
    subtitle: str = "",
    fig_w: float = 16.0,
    fig_h: float = 9.0,
) -> str:
    """Erzeugt PNG + SVG eines Ablaufdiagramms.

    boxes: Liste von dicts:
        label    (Haupttext, ggf. mit \\n)
        sub      (kleiner Untertext, optional)
        color    (Schluessel aus PALETTE)
        x, y, w, h   normalisierte Koordinaten (0..1), y = MITTE der Box
    arrows: Liste von (id_a, id_b) oder (id_a, id_b, label)
    Liefert den PNG-Pfad zurueck.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path

    ids = {b["id"]: b for b in boxes}

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------- Titel ----------
    if title:
        ax.text(0.5, 0.965, title, ha="center", va="top",
                color="#243b53", fontsize=17, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.925, subtitle, ha="center", va="top",
                color="#5c7080", fontsize=10)

    # ---------- Pfeile (unter den Boxen zeichnen) ----------
    for arrow in arrows:
        a, b = arrow[0], arrow[1]
        label = arrow[2] if len(arrow) > 2 else None
        ba, bb = ids[a], ids[b]
        # Box-Kantenpunkte: vom Zentrum der Quellbox zur Zielbox
        ca = (ba["x"], ba["y"])            # Zentrum-Quelle
        cb = (bb["x"], bb["y"])            # Zentrum-Ziel
        # Verbindungslinie kuerzen, damit sie an der Box-Kante endet
        dx, dy = cb[0] - ca[0], cb[1] - ca[1]
        length = max((dx ** 2 + dy ** 2) ** 0.5, 1e-9)
        start = (ca[0] + dx / length * ba["w"] * 0.55,
                 ca[1] + dy / length * ba["h"] * 0.55)
        end = (cb[0] - dx / length * bb["w"] * 0.55,
               cb[1] - dy / length * bb["h"] * 0.55)
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14,
            color="#8aa5bd", lw=1.8, shrinkA=0, shrinkB=0))
        if label:
            mx = (start[0] + end[0]) / 2
            my = (start[1] + end[1]) / 2
            ax.text(mx, my, label, ha="center", va="center",
                    color="#5c7080", fontsize=8.5, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff",
                              ec="#d5dfe7", lw=0.6))

    # ---------- Boxen ----------
    for b in boxes:
        edge, fill = PALETTE[b.get("color", "grau")]
        box = FancyBboxPatch(
            (b["x"] - b["w"] / 2, b["y"] - b["h"] / 2),
            b["w"], b["h"],
            boxstyle="round,pad=0,rounding_size=0.012",
            fc=fill, ec=edge, lw=1.6, zorder=3)
        ax.add_patch(box)
        label = b.get("label", "")
        sub = b.get("sub", "")
        y_off = 0.008 if sub else 0.0
        ax.text(b["x"], b["y"] + y_off + (0.012 if sub else 0),
                label, ha="center", va="center",
                color="#243b53", fontsize=11.5, fontweight="bold", zorder=5)
        if sub:
            ax.text(b["x"], b["y"] - 0.045, sub, ha="center", va="center",
                    color="#5c7080", fontsize=8, zorder=5)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="#ffffff")
    svg_path = output_path.rsplit(".", 1)[0] + ".svg"
    fig.savefig(svg_path, dpi=150, facecolor="#ffffff")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    boxes = [
        {"id": 0, "label": "Start", "color": "blau",
         "x": 0.12, "y": 0.5, "w": 0.16, "h": 0.30},
        {"id": 1, "label": "Verarbeitung", "sub": "Modell X", "color": "lila",
         "x": 0.45, "y": 0.5, "w": 0.20, "h": 0.36},
        {"id": 2, "label": "OK", "color": "gruen",
         "x": 0.80, "y": 0.65, "w": 0.14, "h": 0.26},
        {"id": 3, "label": "Fehler", "color": "rot",
         "x": 0.80, "y": 0.20, "w": 0.14, "h": 0.26},
    ]
    arrows = [(0, 1), (1, 2, "ja"), (1, 3, "nein")]
    p = visualize_flow(boxes, arrows,
                       output_path=os.path.join(
                           os.path.dirname(os.path.abspath(__file__)),
                           "bilder", "flow_beispiel.png"),
                       title="Beispiel: Ablaufdiagramm",
                       subtitle="Boxen + Pfeile mit Labeln")
    print("PNG:", p)