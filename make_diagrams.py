"""Erzeugt die Netzwerk-Diagramme der Dokumentation.

Nutzt den generischen Visualisierer aus visualize_network.py. Rueckruf jederzeit:

    python docs/make_diagrams.py

Erzeugt anschliessend:
    docs/bilder/gesichts_netzwerk.svg   (Gesichtserkennung InsightFace)
    docs/bilder/mlp_beispiel.svg        (einfacher MLP-Demo)
"""

import os
import random

from visualize_network import visualize_network, visualize_network_png

HERE = os.path.dirname(os.path.abspath(__file__))
BILDER = os.path.join(HERE, "bilder")


def _random_weights(sizes, seed=7):
    rnd = random.Random(seed)
    weights = []
    for k in range(len(sizes) - 1):
        n1, n2 = sizes[k], sizes[k + 1]
        weights.append([[rnd.uniform(-1.0, 1.0) for _ in range(n2)]
                        for _ in range(n1)])
    return weights


def _random_activations(sizes, seed=11):
    rnd = random.Random(seed)
    return [[rnd.uniform(0.0, 1.0) for _ in range(n)] for n in sizes]


def main() -> int:
    # ------------------------------------------------------------------
    # 1) Gesichtserkennung — InsightFace buffalo_l
    #    Layerstruktur analog zum Netz: Bild-Merkmale -> Conv -> Embedding
    # ------------------------------------------------------------------
    sizes = [512, 256, 128, 64, 512]
    names = ["INPUT", "HIDDEN 1", "HIDDEN 2", "HIDDEN 3", "OUTPUT"]
    subs = ["112x112x3", "Conv", "Deep", "Dense", "Embedding L2"]
    title = ("Gesichtserkennung — InsightFace")
    sub = ("512 -> 256 -> 128 -> 64 -> 512 Neuronen")
    w = _random_weights([512, 256, 128, 64, 512], seed=3)
    a = _random_activations([512, 256, 128, 64, 512], seed=9)
    p1 = visualize_network(
        layer_sizes=sizes,
        layer_names=names,
        layer_subtitles=subs,
        weights=w,
        activations=a,
        output_path=os.path.join(BILDER, "gesichts_netzwerk.svg"),
        title=title,
        subtitle=sub,
        max_nodes=12,
    )
    print("Gesichtserkennung:", p1)

    # PNG-Version (matplotlib) — gleiche Daten
    p1p = visualize_network_png(
        layer_sizes=sizes,
        layer_names=names,
        layer_subtitles=subs,
        weights=w,
        activations=a,
        output_path=os.path.join(BILDER, "gesichts_netzwerk.png"),
        title=title,
        subtitle=sub,
        max_nodes=12,
    )
    print("Gesichtserkennung (PNG):", p1p)

    # ------------------------------------------------------------------
    # 2) Einfaches MLP-Beispiel (wie im Nutzer-Vorbild)
    # ------------------------------------------------------------------
    demo_sizes = [6, 8, 8, 8, 2]
    p2 = visualize_network(
        layer_sizes=demo_sizes,
        layer_names=["INPUT", "HIDDEN 1", "HIDDEN 2", "HIDDEN 3", "OUTPUT"],
        layer_subtitles=["x1..x6", "ReLU", "ReLU", "ReLU", "Softmax"],
        weights=_random_weights(demo_sizes, seed=5),
        activations=_random_activations(demo_sizes, seed=5),
        output_path=os.path.join(BILDER, "mlp_beispiel.svg"),
        title="Beispiel: MLP",
        subtitle="Neuronen, Gewichte, Pfeile",
        max_nodes=8,
    )
    print("MLP-Beispiel:", p2)

    p2p = visualize_network_png(
        layer_sizes=demo_sizes,
        layer_names=["INPUT", "HIDDEN 1", "HIDDEN 2", "HIDDEN 3", "OUTPUT"],
        layer_subtitles=["x1..x6", "ReLU", "ReLU", "ReLU", "Softmax"],
        weights=_random_weights(demo_sizes, seed=5),
        activations=_random_activations(demo_sizes, seed=5),
        output_path=os.path.join(BILDER, "mlp_beispiel.png"),
        title="Beispiel: MLP",
        subtitle="Neuronen, Gewichte, Pfeile",
    )
    print("MLP-Beispiel (PNG):", p2p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())