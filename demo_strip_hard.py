"""Step 5: a harder wire-strip simulation. Run with: python demo_strip_hard.py   (takes about two minutes)"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src.defect_detector import WireDefectInspector
from src.strip import (DEFECT_TYPES, evaluate_strips, inspect_strip, make_strip, make_wire, match_detections,
                       wander_offset)

N_STRIPS = 20  # strips per cell; each has 12 defects
SIGMA = 3  # Gaussian sensor noise
WIRE_SETTINGS = dict(threshold=20, align="ecc", lock_x=True)


def cell(**kwargs):
    r = evaluate_strips(N_STRIPS, **kwargs)
    return r, f"{r['missed']:>3d} | {r['false_pos']:>3d} | {r['wrong']:>3d}"


def scenarios():
    print(f"1) INCREASING DIFFICULTY  ({N_STRIPS} strips x 12 defects, Gaussian noise sigma={SIGMA})")
    print("   cell: missed | false positives | misclassified")
    rows = [
        ("A  fixed-size defects, straight wire", dict(threshold=20), {}),
        ("B  + random sizes, brightness, angles", dict(threshold=20), dict(varied=True)),
        ("C  + wire wander 5 px, no alignment", dict(threshold=20), dict(varied=True, wander=5)),
        ("D  + wander 5 px, ECC alignment", WIRE_SETTINGS, dict(varied=True, wander=5)),
        ("E  + scratches crossing a wire edge", WIRE_SETTINGS, dict(varied=True, wander=5, edge_scratches=True)),
    ]
    for name, settings, kw in rows:
        r, text = cell(settings=settings, noise_sigma=SIGMA, **kw)
        print(f"   {name:40s}{text}")
        if name.startswith("E"):
            mistakes = {f"{a} -> {b}": n for (a, b), n in r["confusion"].items() if a != b}
            print(f"   misclassified in E: {mistakes}")


def why_defects_are_missed():
    print("\n2) WHICH DEFECTS ARE MISSED (scenario B, varied defects, straight wire)")
    reference = make_wire(6000).astype(int)
    inspector = WireDefectInspector(make_wire(), threshold=20)
    changed = {"found": [], "missed": []}
    per_type = {k: [0, 0] for k in DEFECT_TYPES}  # found, total
    for seed in range(N_STRIPS):
        strip, truth = make_strip(seed=seed, noise_sigma=SIGMA, varied=True)
        clean, _ = make_strip(seed=seed, varied=True)  # same defects without noise, to measure their size
        pairs, _ = match_detections(inspect_strip(strip, inspector), truth)
        for t, d in pairs:
            x, y, w, h = t["box"]
            n_pixels = int((np.abs(clean[y:y + h, x:x + w].astype(int) - reference[y:y + h, x:x + w]).max(axis=2) > 0).sum())
            changed["missed" if d is None else "found"].append(n_pixels)
            per_type[t["type"]][1] += 1
            per_type[t["type"]][0] += d is not None
    print("   found / total:", {k: f"{f}/{t}" for k, (f, t) in per_type.items()})
    print(f"   pixels changed by a missed defect : median {np.median(changed['missed']):.0f}, max {max(changed['missed'])}")
    print(f"   pixels changed by a found defect  : median {np.median(changed['found']):.0f}, min {min(changed['found'])}")
    print("   -> the misses are the tiniest defects (thin 1 px scratches, radius-3 pinholes): a detection floor, not a contrast problem")


def settings_tradeoff():
    print(f"\n3) THE TRADE-OFF: catching small defects vs. surviving speck noise  (scenario B, {N_STRIPS * 12} defects)")
    print("   cell: missed | false positives")
    print("   threshold sweep (blur 5x5, 3x3 open, min area 20):")
    print(f"   {'threshold':>10s}" + "".join(f"{f'sigma={s}':>14s}" for s in (0, 6)))
    for thr in (10, 15, 20, 30, 40):
        row = f"   {thr:>10d}"
        for s in (0, 6):
            r = evaluate_strips(N_STRIPS, dict(threshold=thr), noise_sigma=s, varied=True)
            row += f"{r['missed']:>8d} | {r['false_pos']:<3d}"
        print(row)
    print("   smoothing / cleanup (threshold 20), with salt-and-pepper specks (S&P):")
    conditions = [("gauss s=3", dict(noise_sigma=3)), ("S&P 1%", dict(speck_density=0.01)),
                  ("S&P 3%", dict(speck_density=0.03)), ("S&P 6%", dict(speck_density=0.06))]
    print(f"   {'':38s}" + "".join(f"{name:>13s}" for name, _ in conditions))
    configs = [
        ("current: blur 5, open 3x3, min 20", dict(threshold=20)),
        ("no open, blur 3, min 8", dict(threshold=20, blur_kernel=(3, 3), morph_kernel=(1, 1), min_defect_size=8)),
        ("no open, blur 3, min 8 + median", dict(threshold=20, blur_kernel=(3, 3), morph_kernel=(1, 1), min_defect_size=8,
                                                 denoise="median+gaussian")),
        ("current, min 8 + median", dict(threshold=20, min_defect_size=8, denoise="median+gaussian")),
    ]
    for name, settings in configs:
        row = f"   {name:38s}"
        for _, kw in conditions:
            r = evaluate_strips(N_STRIPS, settings, varied=True, **kw)
            row += f"{r['missed']:>7d} | {r['false_pos']:<4d}"
        print(row)


def position_offsets():
    print("\n4) POSITION ERROR OF DETECTIONS vs. ground truth, wire wander 5 px (centre offsets in px)")
    for name, settings in [("ECC alignment", dict(threshold=20, align="ecc")),
                           ("ECC alignment, lock_x=True", WIRE_SETTINGS)]:
        inspector = WireDefectInspector(make_wire(), **settings)
        dx = []
        for seed in range(N_STRIPS):
            strip, truth = make_strip(seed=seed, noise_sigma=SIGMA, varied=True, wander=5)
            found = inspect_strip(strip, inspector)
            for t in truth:
                tx, ty, tw, th = t["box"]
                near = [d for d in found if abs(d["x"] + d["w"] / 2 - tx - tw / 2) < 20 and abs(d["y"] + d["h"] / 2 - ty - th / 2) < 20]
                if near:
                    d = min(near, key=lambda d: abs(d["x"] + d["w"] / 2 - tx - tw / 2))
                    dx.append(abs(d["x"] + d["w"] / 2 - tx - tw / 2))
        print(f"   {name:30s} along the wire (x): median {np.median(dx):.1f}, 90th percentile {np.percentile(dx, 90):.1f}, max {max(dx):.1f}")


def draw_strip(seed=12):
    strip, truth = make_strip(seed=seed, noise_sigma=SIGMA, varied=True, wander=5, edge_scratches=True)
    inspector = WireDefectInspector(make_wire(), **WIRE_SETTINGS)
    found = inspect_strip(strip, inspector)
    pairs, extra = match_detections(found, truth)
    # detections are in the straight (aligned) frame; shift boxes by the local wander so they sit on the drifting wire
    shift = wander_offset(seed, strip.shape[1], 5)
    seg = 1500
    fig, axs = plt.subplots(4, 1, figsize=(16, 8))
    for i, ax in enumerate(axs):
        x0 = i * seg
        ax.imshow(cv2.cvtColor(strip[:, x0:x0 + seg], cv2.COLOR_BGR2RGB))
        ax.set_yticks([])
        ax.set_xticks(range(0, seg + 1, 250), [str(x0 + v) for v in range(0, seg + 1, 250)], fontsize=7)

        def box(b, color, label, style="-"):
            x, y, w, h = b
            dy = shift[min(max(x + w // 2, 0), len(shift) - 1)]
            ax.add_patch(Rectangle((x - x0 - 1.5, y + dy - 1.5), w + 3, h + 3, fill=False, ec=color, lw=1.5, ls=style))
            ax.text(x - x0, y + dy - 4, label, color=color, fontsize=7.5)

        for t, d in pairs:
            tb = t["box"]
            if not x0 <= tb[0] < x0 + seg:
                continue
            if d is None:
                box(tb, "white", "missed", "--")
            elif d["type"] == t["type"]:
                box((d["x"], d["y"], d["w"], d["h"]), "lime", d["type"])
            else:
                box((d["x"], d["y"], d["w"], d["h"]), "red", f"{d['type']} (true: {t['type']})")
        for d in extra:
            if x0 <= d["x"] < x0 + seg:
                box((d["x"], d["y"], d["w"], d["h"]), "red", f"false: {d['type']}", ":")
    fig.suptitle("Wandering wire with varied defects. green = right type, red = wrong type / false positive, white dashed = missed", fontsize=10)
    fig.tight_layout()
    fig.savefig("demo_strip_hard.png", dpi=100)


def main():
    scenarios()
    why_defects_are_missed()
    settings_tradeoff()
    position_offsets()
    draw_strip()
    plt.show()


if __name__ == "__main__":
    main()
