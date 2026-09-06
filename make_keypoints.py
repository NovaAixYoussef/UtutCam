"""Erzeugt das COCO-17-Keypoints-Skelett-Diagramm der Dieberkennung.

    python docs/make_keypoints.py

Erzeugt in docs/bilder/:
    dieb_keypoints.png|svg   COCO-17-Body-Keypoints (YOLOv8n-pose)

Die 17 Punkte und Knochen entsprechen dem in dieb_erkennung/pose_detector.py
verwendeten COCO-17-Modell (yolov8n-pose).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
BILDER = os.path.join(HERE, "bilder")

# COCO-17 Keypoints
KPS = [
    ("0  Nase", (0.0, 1.55)),
    ("1  Li. Auge", (-0.16, 1.72)),
    ("2  Re. Auge", (0.16, 1.72)),
    ("3  Li. Ohr", (-0.30, 1.70)),
    ("4  Re. Ohr", (0.30, 1.70)),
    ("5  Li. Schulter", (-0.34, 1.25)),
    ("6  Re. Schulter", (0.34, 1.25)),
    ("7  Li. Ellbogen", (-0.50, 0.85)),
    ("8  Re. Ellbogen", (0.50, 0.85)),
    ("9  Li. Hand", (-0.55, 0.42)),
    ("10 Re. Hand", (0.55, 0.42)),
    ("11 Li. Huefte", (-0.28, 0.55)),
    ("12 Re. Huefte", (0.28, 0.55)),
    ("13 Li. Knie", (-0.28, 0.05)),
    ("14 Re. Knie", (0.28, 0.05)),
    ("15 Li. Fuss", (-0.26, -0.35)),
    ("16 Re. Fuss", (0.26, -0.35)),
]

# Knochen (Paar-IDs)
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # Kopf
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arme
    (5, 11), (6, 12), (11, 12),          # Rumpf
    (11, 13), (13, 15), (12, 14), (14, 16),   # Beine
]

POINT_COLOR = "#5aa7de"     # Brand-Blau
BONE_COLOR = "#9d87c6"      # Brand-Lila
BG = "#ffffff"
TEXT = "#243b53"
DIM = "#5c7080"


def draw():
    fig, ax = plt.subplots(figsize=(7.2, 9.0), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Knochen
    for a, b in BONES:
        (x1, y1), (x2, y2) = KPS[a][1], KPS[b][1]
        ax.plot([x1, x2], [y1, y2], color=BONE_COLOR, lw=3.2,
                solid_capstyle="round", zorder=2)

    # Punkte + Beschriftung
    for name, (x, y) in KPS:
        ax.scatter(x, y, s=520, c=POINT_COLOR, edgecolors="white",
                   linewidths=1.6, zorder=3)
        dx = -0.16 if x <= 0 else 0.16
        ha = "right" if x <= 0 else "left"
        ax.text(x + dx, y, name, fontsize=10.5, color=TEXT, ha=ha,
                va="center", zorder=4)

    ax.set_xlim(-1.05, 1.15)
    ax.set_ylim(-0.55, 2.05)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title("COCO-17 Body-Keypoints\n(yolov8n-pose, TheftGuard / Pose-Detektor)",
                 fontsize=13, color=TEXT, pad=16, fontweight="bold")

    # Kopf-Titellegende
    ax.text(0.0, -0.52,
            "17 Punkte: Nase, Augen, Ohren, Schultern, Ellbogen,\n"
            "Handgelenke, Hueften, Knie, Fuesse",
            fontsize=9.5, color=DIM, ha="center", va="top", zorder=4)

    fig.tight_layout()
    png = os.path.join(BILDER, "dieb_keypoints.png")
    svg = os.path.join(BILDER, "dieb_keypoints.svg")
    fig.savefig(png, bbox_inches="tight", facecolor=BG)
    fig.savefig(svg, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return png, svg


if __name__ == "__main__":
    p, s = draw()
    print("keypoints:", p, "|", s)
