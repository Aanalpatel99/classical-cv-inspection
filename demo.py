import cv2
import matplotlib.pyplot as plt

from src.defect_detector import WireDefectInspector
from src.utils import make_reference, make_test


def show(ax, img_bgr, title):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis("off")


def main():
    ref = make_reference()
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

    plt.tight_layout()
    plt.savefig("demo_output.png", dpi=100)
    plt.show()


if __name__ == "__main__":
    main()
