"""Step 4: simulated line-scan wire inspection. Run with: python demo_strip.py"""
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.defect_detector import WireDefectInspector
from src.strip import CHUNK_W, DEFECT_TYPES, OVERLAP, inspect_strip, make_strip, make_wire, match_detections

COLORS = {"pinhole": "cyan", "scratch": "yellow", "lump": "magenta", "neckdown": "orange", "unknown": "red"}
# threshold 20, not the default 30: 2px diagonal scratches are faint after blurring (see the evaluation below)
STRIP_SETTINGS = dict(threshold=20)


def evaluate(n_strips, sigma, settings):
    """Run n_strips random strips; return (missed, false positives, confusion counts, features per true type)."""
    inspector = WireDefectInspector(make_wire(), **settings)
    missed = false_pos = 0
    confusion = defaultdict(int)
    features = defaultdict(list)
    for seed in range(n_strips):
        strip, truth = make_strip(seed=seed, noise_sigma=sigma)
        pairs, extra = match_detections(inspect_strip(strip, inspector), truth)
        false_pos += len(extra)
        for t, d in pairs:
            if d is None:
                missed += 1
                continue
            confusion[(t["type"], d["type"])] += 1
            features[t["type"]].append([d["area"], d["circularity"], d["aspect_ratio"], d["polarity"]])
    return missed, false_pos, confusion, features


def main():
    n_defects = 12
    inspector = WireDefectInspector(make_wire(), **STRIP_SETTINGS)
    strip, truth = make_strip(seed=0)
    found = inspect_strip(strip, inspector)
    pairs, extra = match_detections(found, truth)

    print(f"STRIP: {strip.shape[1]} x {strip.shape[0]} px, inspected as chunks of {CHUNK_W} px overlapping by {OVERLAP} px")
    print(f"{'true type':10s}{'true box (x, y, w, h)':>26s}   {'classified as':14s}")
    for t, d in pairs:
        print(f"{t['type']:10s}{str(t['box']):>26s}   {d['type'] if d else 'MISSED':14s}{'' if not d or d['type'] == t['type'] else '  <-- wrong'}")
    print(f"false positives: {len(extra)}")

    seg = 1500
    fig, axs = plt.subplots(4, 1, figsize=(16, 7.5))
    for i, ax in enumerate(axs):
        x0 = i * seg
        ax.imshow(cv2.cvtColor(strip[:, x0:x0 + seg], cv2.COLOR_BGR2RGB))
        ax.set_yticks([])
        ax.set_xticks(range(0, seg + 1, 250), [str(x0 + v) for v in range(0, seg + 1, 250)], fontsize=7)
        for d in found:
            if x0 <= d["x"] < x0 + seg:
                ax.add_patch(Rectangle((d["x"] - x0 - 1.5, d["y"] - 1.5), d["w"] + 3, d["h"] + 3, fill=False, ec=COLORS[d["type"]], lw=1.5))
                ax.text(d["x"] - x0, d["y"] - 4, d["type"], color=COLORS[d["type"]], fontsize=8, ha="left")
    fig.suptitle("Simulated wire strip (6000 px), defects found and classified", fontsize=10)
    fig.tight_layout()
    fig.savefig("demo_strip.png", dpi=100)

    n_strips = 30
    _, _, confusion, features = evaluate(n_strips, 0, STRIP_SETTINGS)
    print(f"\nFEATURES the classifier sees (min .. max over {sum(len(v) for v in features.values())} detected defects)")
    print(f"{'true type':10s}{'area':>16s}{'circularity':>16s}{'aspect ratio':>16s}{'polarity':>18s}")
    for k in DEFECT_TYPES:
        a = np.array(features[k])
        print(f"{k:10s}" + "".join(f"{f'{a[:, i].min():.2f}..{a[:, i].max():.2f}':>{w}s}" for i, w in enumerate([16, 16, 16, 18])))
    print("  polarity > 0: brighter than the reference (extra material); < 0: darker (missing material)")

    print(f"\nEVALUATION: {n_strips} random strips x {n_defects} defects = {n_strips * n_defects} true defects, sensor noise added to every pixel")
    print("  cell: missed | false positives | misclassified")
    settings = {"default settings": {}, "threshold=20": STRIP_SETTINGS}
    print(f"{'noise sigma':>12s}" + "".join(f"{name:>22s}" for name in settings))
    for sigma in [0, 3, 6, 10]:
        row = f"{sigma:>12d}"
        for kwargs in settings.values():
            missed, false_pos, conf, _ = evaluate(n_strips, sigma, kwargs)
            wrong = sum(n for (true, pred), n in conf.items() if true != pred)
            row += f"{f'{missed} | {false_pos} | {wrong}':>22s}"
        print(row)

    print("\nCONFUSION at noise sigma 10, threshold=20 (rows: true type, columns: classified as)")
    _, _, conf, _ = evaluate(n_strips, 10, STRIP_SETTINGS)
    print(f"{'':10s}" + "".join(f"{t:>10s}" for t in DEFECT_TYPES))
    for true in DEFECT_TYPES:
        print(f"{true:10s}" + "".join(f"{conf[(true, pred)]:>10d}" for pred in DEFECT_TYPES))
    plt.show()


if __name__ == "__main__":
    main()
