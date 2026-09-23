import cv2
import matplotlib.pyplot as plt

from src.defect_detector import WireDefectInspector
from src.utils import apply_lighting_gradient, make_reference, make_test


def show(ax, img_bgr, title):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def basic_demo(ref):
    """Clean / noise-only / defective parts under the reference lighting."""
    inspector = WireDefectInspector(ref)

    cases = {
        "Clean part": make_test(ref, big_defect=False, thin_line=False, short_line=False, noise_dots=False),
        "Noise only": make_test(ref, big_defect=False, thin_line=False, short_line=False),
        "Circle + 2 lines + noise": make_test(ref),
    }

    fig, axs = plt.subplots(len(cases), 2, figsize=(10, 3 * len(cases)))
    for row, (name, test) in enumerate(cases.items()):
        result = inspector.inspect(test)
        print(f"\n{name}: {result['status']} ({len(result['defects'])} defect(s))")
        for d in result["defects"]:
            print(f"  area={d['area']:.0f}  box=({d['x']}, {d['y']}, {d['w']}, {d['h']})")
        show(axs[row][0], test, f"{name} - test image")
        show(axs[row][1], result["defect_img"], f"Result: {result['status']}")

    fig.tight_layout()
    fig.savefig("demo_output.png", dpi=100)


def lighting_demo(ref, left=0.7, right=1.3):
    """Same parts, but lit unevenly. Which preprocessing keeps the clean part clean and still finds the defects?"""
    methods = {
        "fixed threshold": dict(),
        "Otsu": dict(threshold_mode="otsu"),
        "adaptive": dict(threshold_mode="adaptive"),
        "fixed + CLAHE": dict(use_clahe=True),
        "fixed + illumination\nnormalization": dict(normalize_illumination=True),
    }
    parts = {
        "clean": apply_lighting_gradient(make_test(ref, False, False, False, False), left, right),
        "defective": apply_lighting_gradient(make_test(ref), left, right),
    }

    print(f"\nLighting gradient {left}-{right}  (real defects in the defective part: 3)")
    print(f"{'method':38s}{'clean part':>12s}{'defective part':>16s}")
    fig, axs = plt.subplots(len(parts), len(methods), figsize=(4 * len(methods), 2.6 * len(parts)))
    for col, (name, kwargs) in enumerate(methods.items()):
        inspector = WireDefectInspector(ref, **kwargs)
        found = {}
        for row, (part, img) in enumerate(parts.items()):
            result = inspector.inspect(img)
            found[part] = len(result["defects"])
            show(axs[row][col], result["defect_img"], f"{name.replace(chr(10), ' ')}\n{part}: {found[part]} defect(s)")
        print(f"{name.replace(chr(10), ' '):38s}{found['clean']:>12d}{found['defective']:>16d}")

    fig.tight_layout()
    fig.savefig("demo_lighting.png", dpi=100)


def main():
    ref = make_reference()
    basic_demo(ref)
    lighting_demo(ref)
    plt.show()


if __name__ == "__main__":
    main()
