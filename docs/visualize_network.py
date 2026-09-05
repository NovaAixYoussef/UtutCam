"""Automatischer Neuronales-Netzwerk-Visualisierer (Light-Mode).

Zeichnet aus einer einfachen Layer-Definition eine professionelle SVG-Grafik
wie in der klassischen Netzwerkdarstellung:

    Input  ->  Hidden 1  ->  Hidden 2  ->  Hidden 3  ->  Output
      o          o             o             o             o
      o -------  o --------    o --------    o --------    o
      o          o             o             o             o

Design: heller Hintergrund, gedeckte Farben (kein Neon / kein Dark-Mode),
klare Abstaende, kurze Beschriftungen.

Kernfunktion::

    visualize_network(
        layer_sizes  = [784, 128, 64, 32, 10],
        layer_names  = ["INPUT", "HIDDEN 1", "HIDDEN 2", "HIDDEN 3", "OUTPUT"],
        title        = "Mein Netz",
    )

Optionale Features (alles weglassbar):
- weights: Liste von Gewichtsmatrizen -> Linienstaerke variiert je Gewicht
- activations: Liste von Aktivierungsvektoren -> Neuronen-Groesse
- max_nodes: begrenzt gezeichnete Neuronen pro Layer (nur Anzeige-Abtastung),
  die echte Anzahl wird trotzdem als Text angegeben
- Pfeile zwischen den Schichten, Layer-Namen, Neuronen-Anzahl, Titel, Farben
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Standard-Farben (hell, gedeckt)
# ---------------------------------------------------------------------------
COL_BG1 = "#ffffff"
COL_BG2 = "#f2f6fa"
COL_INPUT = "#5aa7de"
COL_HIDDEN = "#9d87c6"
COL_OUTPUT = "#5bb88a"
COL_LINE = "#8aa5bd"
COL_TEXT = "#243b53"
COL_TEXT_DIM = "#5c7080"


def _bg_gradient():
    return (f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{COL_BG1}"/>'
            f'<stop offset="100%" stop-color="{COL_BG2}"/>'
            f"</linearGradient>")


def _node_colors(layer_idx: int, total: int):
    if layer_idx == 0:
        c1, c2 = "#5aa7de", "#7ebbe6"
    elif layer_idx == total - 1:
        c1, c2 = "#5bb88a", "#7cc9a3"
    else:
        c1, c2 = "#9d87c6", "#b3a1d6"
    gid = f"node{layer_idx}"
    return (f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0%" stop-color="{c1}"/>'
            f'<stop offset="100%" stop-color="{c2}"/>'
            f"</linearGradient>"), gid


def _soft_shadow():
    return (f'<filter id="sh" x="-40%" y="-40%" width="180%" height="180%">'
            f'<feDropShadow dx="0" dy="1" stdDeviation="1.5" '
            f'flood-color="#243b53" flood-opacity="0.15"/>'
            f"</filter>")


def visualize_network(
    layer_sizes: Sequence[int],
    layer_names: Optional[Sequence[str]] = None,
    layer_subtitles: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    activations: Optional[Sequence[Sequence[float]]] = None,
    output_path: str = "netzwerk.svg",
    title: str = "Neuronales Netzwerk",
    subtitle: str = "",
    max_nodes: int = 12,
    show_neuron_count: bool = True,
    show_arrow_layer: bool = True,
    line_alpha: float = 0.22,
) -> str:
    """Erzeugt die SVG-Grafik und liefert den Pfad zurueck."""
    n_layers = len(layer_sizes)
    if layer_names is None:
        layer_names = [f"LAYER {i}" for i in range(n_layers)]
    if layer_subtitles is None:
        layer_subtitles = [""] * n_layers
    if weights is None:
        weights = [None] * (n_layers - 1)
    if activations is None:
        activations = [None] * n_layers

    # ---------- Layout-Rechnung ----------
    MARGIN_L, MARGIN_R = 95, 95
    TOP = 96
    LAYER_GAP = 235
    NODE_SPACING = 32
    NODE_R = 8

    max_nodes = min(max_nodes, max(layer_sizes))
    width = MARGIN_L + LAYER_GAP * (n_layers - 1) + MARGIN_R
    heights = [max(90, (min(s, max_nodes) - 1) * NODE_SPACING + 2 * NODE_R)
               for s in layer_sizes]
    height = max(heights) + TOP + 210

    centers = [MARGIN_L + LAYER_GAP * i for i in range(n_layers)]

    # ------------------------- SVG-Aufbau -------------------------
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                 f'height="{height}" viewBox="0 0 {width} {height}">')
    arrow_marker = ('<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" '
                    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{COL_LINE}"/>'
                    "</marker>")
    parts.append("<defs>" + _bg_gradient())
    for i in range(n_layers):
        parts.append(_node_colors(i, n_layers)[0])
    parts.append(arrow_marker + _soft_shadow() + "</defs>")
    parts.append(f'<rect width="{width}" height="{height}" fill="url(#bg)"/>')

    # Titel + Untertitel
    parts.append(f'<text x="{width/2:.0f}" y="34" text-anchor="middle" '
                 f'font-family="Arial" font-size="21" font-weight="bold" '
                 f'fill="{COL_TEXT}">{_esc(title)}</text>')
    if subtitle:
        parts.append(f'<text x="{width/2:.0f}" y="56" text-anchor="middle" '
                     f'font-family="Arial" font-size="12" '
                     f'fill="{COL_TEXT_DIM}">{_esc(subtitle)}</text>')

    # Koordinaten der Layer-Graphen
    coords = []
    for i, s in enumerate(layer_sizes):
        cy = TOP + 55
        n_show = min(s, max_nodes)
        ys = []
        for k in range(n_show):
            off = (n_show - 1) / 2.0
            y = cy + (k - off) * NODE_SPACING
            ys.append(y)
        coords.append((centers[i], ys))

    # ---------- Verbindungen (Liniendicken je nach Gewicht) ----------
    for li in range(n_layers - 1):
        cx1, ys1 = coords[li]
        cx2, ys2 = coords[li + 1]
        wm = weights[li]
        for i1, y1 in enumerate(ys1):
            for j, y2 in enumerate(ys2):
                alpha = line_alpha
                sw = 1.0
                if wm is not None:
                    try:
                        sz = abs(float(wm[i1][j]))
                    except Exception:
                        sz = 0.5
                    sw = 0.4 + 1.4 * min(sz, 1.0)
                    alpha = min(0.45, 0.08 + 0.37 * min(sz, 1.0))
                parts.append(
                    f'<line x1="{cx1:.0f}" y1="{y1:.0f}" x2="{cx2:.0f}" '
                    f'y2="{y2:.0f}" stroke="#7e97ad" stroke-width="{sw:.2f}" '
                    f'opacity="{alpha:.2f}"/>')

    # ---------- Neuronen ----------
    for i in range(n_layers):
        cx, ys = coords[i]
        n_show = len(ys)
        _, gid = _node_colors(i, n_layers)
        for k, y in enumerate(ys):
            act = activations[i]
            fill = f"url(#{gid})"
            radius = NODE_R
            extra = ""
            if act is not None and k < len(act):
                v = max(0.0, min(1.0, float(act[k])))
                radius = NODE_R * (0.6 + 0.4 * v)
                extra = f' fill-opacity="{(0.55 + 0.45 * v):.2f}"'
            parts.append(f'<circle cx="{cx:.0f}" cy="{y:.0f}" '
                         f'r="{radius}" fill="{fill}"{extra} '
                         f'filter="url(#sh)" stroke="#ffffff" stroke-width="0.6"/>')

        # Mehr-Knoten-Hinweis
        if layer_sizes[i] > n_show:
            last_y = ys[-1]
            parts.append(f'<text x="{cx:.0f}" y="{last_y + 18:.0f}" '
                         f'text-anchor="middle" font-family="Arial" '
                         f'font-size="11" fill="{COL_TEXT_DIM}">(… '
                         f'{layer_sizes[i] - n_show} weitere)</text>')

        # Layer-Name (kurz)
        parts.append(f'<text x="{cx:.0f}" y="{TOP}" text-anchor="middle" '
                     f'font-family="Arial" font-size="13" font-weight="bold" '
                     f'fill="{COL_TEXT}">{_esc(layer_names[i])}</text>')
        if layer_subtitles[i]:
            parts.append(f'<text x="{cx:.0f}" y="{TOP + 15}" '
                         f'text-anchor="middle" font-family="Arial" '
                         f'font-size="10.5" fill="{COL_TEXT_DIM}">'
                         f'{_esc(layer_subtitles[i])}</text>')

        # Neuronen-Anzahl (unter der Schicht)
        if show_neuron_count:
            base = coords[i][1][-1] + 42 if ys else TOP + 55 + 42
            parts.append(f'<text x="{cx:.0f}" y="{base:.0f}" '
                         f'text-anchor="middle" font-family="Arial" '
                         f'font-size="11.5" fill="{COL_TEXT_DIM}">'
                         f'{layer_sizes[i]} Neuronen</text>')

    # ---------- Pfeile zwischen den Layern ----------
    if show_arrow_layer:
        for i in range(n_layers - 1):
            cx1 = centers[i]
            cx2 = centers[i + 1]
            midx = (cx1 + cx2) / 2
            parts.append(
                f'<line x1="{midx - 10:.0f}" y1="{TOP + 8}" '
                f'x2="{midx + 10:.0f}" y2="{TOP + 8}" '
                f'stroke="{COL_LINE}" stroke-width="2.5" '
                f'marker-end="url(#arrow)"/>')

    parts.append("</svg>")
    svg = "\n".join(parts)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(svg)
    return output_path


def _esc(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def visualize_network_png(
    layer_sizes: Sequence[int],
    layer_names: Optional[Sequence[str]] = None,
    layer_subtitles: Optional[Sequence[str]] = None,
    weights: Optional[Sequence[Sequence[Sequence[float]]]] = None,
    activations: Optional[Sequence[Sequence[float]]] = None,
    output_path: str = "netzwerk.png",
    title: str = "Neuronales Netzwerk",
    subtitle: str = "",
    max_nodes: int = 12,
    show_arrow_layer: bool = True,
) -> str:
    """Wie visualize_network(), aber als PNG (matplotlib, kein Cairo noetig)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    n_layers = len(layer_sizes)
    if layer_names is None:
        layer_names = [f"LAYER {i}" for i in range(n_layers)]
    if layer_subtitles is None:
        layer_subtitles = [""] * n_layers
    if weights is None:
        weights = [None] * (n_layers - 1)
    if activations is None:
        activations = [None] * n_layers

    layer_colors = ["#5aa7de", "#9d87c6", "#9d87c6", "#9d87c6", "#5bb88a"]
    if n_layers == 2:
        layer_colors = ["#5aa7de", "#5bb88a"]

    fig_w = 5.5 + 2.6 * (n_layers - 1)
    fig, ax = plt.subplots(figsize=(fig_w, 6.2), dpi=150)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs_pos = [0.08 + 0.84 * i / (n_layers - 1) for i in range(n_layers)]
    max_nodes = min(max_nodes, max(layer_sizes))
    for li in range(n_layers - 1):
        xa, xb = xs_pos[li], xs_pos[li + 1]
        na, nb = min(layer_sizes[li], max_nodes), min(layer_sizes[li + 1], max_nodes)
        ya = [0.53 + 0.50 * (k - (na - 1) / 2) / max(na - 1, 1) for k in range(na)]
        yb = [0.53 + 0.50 * (k - (nb - 1) / 2) / max(nb - 1, 1) for k in range(nb)]
        wm = weights[li]
        for i1, y1 in enumerate(ya):
            for j, y2 in enumerate(yb):
                alpha, lw = 0.16, 0.4
                if wm is not None:
                    try:
                        sz = abs(float(wm[i1][j]))
                    except Exception:
                        sz = 0.5
                    lw = 0.2 + 1.4 * min(sz, 1.0)
                    alpha = min(0.42, 0.06 + 0.36 * min(sz, 1.0))
                color = (0.49, 0.59, 0.68, alpha)
                ax.plot([xa, xb], [y1, y2], color=color, lw=lw,
                        solid_capstyle="round")

    if show_arrow_layer:
        for i in range(n_layers - 1):
            x0 = (xs_pos[i] + xs_pos[i + 1]) / 2
            ax.add_patch(FancyArrowPatch((x0 - 0.013, 0.055), (x0 + 0.013, 0.055),
                                         arrowstyle="-|>", mutation_scale=11,
                                         color="#8aa5bd", lw=1.6))

    for i in range(n_layers):
        xx = xs_pos[i]
        n = min(layer_sizes[i], max_nodes)
        ya = [0.53 + 0.50 * (k - (n - 1) / 2) / max(n - 1, 1) for k in range(n)]
        col = layer_colors[min(i, len(layer_colors) - 1)]
        act = activations[i]
        for k, y in enumerate(ya):
            s = 80
            alpha_f = 0.85
            if act is not None and k < len(act):
                v = max(0.0, min(1.0, float(act[k])))
                s = 45 + 90 * v
                alpha_f = 0.55 + 0.45 * v
            ax.scatter(xx, y, s=s, color=col, alpha=alpha_f,
                       edgecolors=(1.0, 1.0, 1.0), linewidths=0.6, zorder=3)

        ax.text(xx, 0.955, layer_names[i], ha="center", va="top",
                color="#243b53", fontsize=12, fontweight="bold")
        if layer_subtitles[i]:
            ax.text(xx, 0.90, layer_subtitles[i], ha="center", va="top",
                    color="#5c7080", fontsize=8)
        ax.text(xx, 0.12, f"{layer_sizes[i]} Neuronen", ha="center", va="bottom",
                color="#5c7080", fontsize=9)

    ax.set_title(title, color="#243b53", fontsize=13, fontweight="bold", pad=14)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", color="#5c7080", fontsize=9)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    demo = {
        "layer_sizes": [6, 8, 8, 8, 2],
        "layer_names": ["INPUT", "HIDDEN 1", "HIDDEN 2", "HIDDEN 3", "OUTPUT"],
        "layer_subtitles": ["x1..x6", "ReLU", "ReLU", "ReLU", "Softmax"],
        "title": "Beispiel: MLP",
        "subtitle": "Neuronen, Gewichte, Pfeile",
        "max_nodes": 8,
    }
    p = visualize_network(output_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bilder", "mlp_beispiel.svg"),
        **demo)
    print("Demo-SVG:", p)
    png = visualize_network_png(output_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bilder", "mlp_beispiel.png"),
        **demo)
    print("Demo-PNG:", png)