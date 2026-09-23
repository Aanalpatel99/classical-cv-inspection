import cv2
import matplotlib.pyplot as plt

from src.defect_detector import WireDefectInspector
from src.utils import (SMALL_DEFECTS, add_salt_and_pepper, apply_lighting_gradient, make_reference,
                       make_test, misalign, score_detections)


def show(ax, img_bgr, title):
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def evaluate(inspector, clean_img, defect_img):
    """Run one inspector on a clean and a defective image; return both results plus scores."""
    clean_result = inspector.inspect(clean_img)
    defect_result = inspector.inspect(defect_img)
    clean_fp = len(clean_result["defects"])  # on a clean part, every detection is a false positive
    found, defect_fp = score_detections(defect_result["defects"])
    return clean_result, defect_result, clean_fp, found, defect_fp


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
    clean_img = apply_lighting_gradient(make_test(ref, False, False, False, False), left, right)
    defect_img = apply_lighting_gradient(make_test(ref), left, right)

    print(f"\nLIGHTING: gradient {left}-{right}")
    print(f"{'method':38s}{'clean: false pos':>18s}{'defective: found':>18s}{'false pos':>11s}")
    fig, axs = plt.subplots(2, len(methods), figsize=(4 * len(methods), 5.2))
    for col, (name, kwargs) in enumerate(methods.items()):
        flat = name.replace("\n", " ")
        c, d, clean_fp, found, defect_fp = evaluate(WireDefectInspector(ref, **kwargs), clean_img, defect_img)
        print(f"{flat:38s}{clean_fp:>18d}{found:>15d}/3{defect_fp:>11d}")
        show(axs[0][col], c["defect_img"], f"{flat}\nclean: {clean_fp} false positive(s)")
        show(axs[1][col], d["defect_img"], f"defective: {found}/3 found, {defect_fp} false pos.")

    fig.tight_layout()
    fig.savefig("demo_lighting.png", dpi=100)


def alignment_demo(ref):
    """Parts that arrive shifted or rotated. Does aligning to the reference first help?"""
    methods = {
        "no alignment": dict(),
        "template matching": dict(align="template"),
        "ECC": dict(align="ecc"),
        "ECC + illumination norm.": dict(align="ecc", normalize_illumination=True),
    }
    # (label, dx, dy, rotation in degrees, lighting gradient or None)
    misalignments = [
        ("shift (3, 4) px", 3, 4, 0, None),
        ("shift (-5, 5) px", -5, 5, 0, None),
        ("subpixel shift (1.5, -2.5)", 1.5, -2.5, 0, None),
        ("rotation 1.5 deg", 0, 0, 1.5, None),
        ("shift (4, -3) + rot 2 deg", 4, -3, 2, None),
        ("shift (4, -3) + rot 2 + uneven light", 4, -3, 2, (0.7, 1.3)),
    ]
    clean_base = make_test(ref, False, False, False, False)
    defect_base = make_test(ref)

    def prepare(img, dx, dy, angle, light):
        img = misalign(img, dx, dy, angle)
        return apply_lighting_gradient(img, *light) if light else img

    print("\nALIGNMENT: real defects found (of 3) / false positives  [clean part false positives]")
    print(f"{'misalignment':40s}" + "".join(f"{name:>26s}" for name in methods))
    hard = {}
    for label, dx, dy, angle, light in misalignments:
        clean_img = prepare(clean_base, dx, dy, angle, light)
        defect_img = prepare(defect_base, dx, dy, angle, light)
        row = f"{label:40s}"
        for name, kwargs in methods.items():
            c, d, clean_fp, found, defect_fp = evaluate(WireDefectInspector(ref, **kwargs), clean_img, defect_img)
            row += f"{f'{found}/3, {defect_fp} fp  [{clean_fp}]':>26s}"
            if label.startswith("shift (4, -3) + rot 2 deg"):
                hard[name] = (c, d, clean_fp, found, defect_fp)
        print(row)

    fig, axs = plt.subplots(2, len(methods), figsize=(4 * len(methods), 5.2))
    for col, (name, (c, d, clean_fp, found, defect_fp)) in enumerate(hard.items()):
        show(axs[0][col], c["defect_img"], f"{name}\nclean: {clean_fp} false positive(s)")
        show(axs[1][col], d["defect_img"], f"defective: {found}/3 found, {defect_fp} false pos.")
    fig.suptitle("Part shifted (4, -3) px and rotated 2 degrees", fontsize=10)
    fig.tight_layout()
    fig.savefig("demo_alignment.png", dpi=100)


def noise_demo(ref, n_patterns=10):
    """Salt-and-pepper noise: when does the default Gaussian pipeline break, and what does a median filter cost?"""
    methods = {
        "gaussian (default)": dict(),
        "median 3x3": dict(denoise="median", min_defect_size=8),
        "median 3x3 + gaussian": dict(denoise="median+gaussian", min_defect_size=8),
    }
    clean_base = make_test(ref, False, False, False, False)
    defect_base = make_test(ref)
    seeds = range(n_patterns)

    print(f"\nNOISE, part 1: density sweep (averages over {n_patterns} random noise patterns)")
    print("  each cell: false positives on a clean part | real defects found (of 3), false positives on a defective part")
    print(f"{'density':>9s}" + "".join(f"{name:>36s}" for name in methods))
    for density in [0.0, 0.02, 0.05, 0.10, 0.15]:
        row = f"{density:>8.0%} "
        for kwargs in methods.values():
            inspector = WireDefectInspector(ref, **kwargs)
            clean_fp = found = defect_fp = 0
            for seed in seeds:
                clean_fp += len(inspector.inspect(add_salt_and_pepper(clean_base, density, seed))["defects"])
                f, fp = score_detections(inspector.inspect(add_salt_and_pepper(defect_base, density, seed))["defects"])
                found += f
                defect_fp += fp
            n = n_patterns
            row += f"{f'{clean_fp / n:.1f} | {found / n:.1f}/3, {defect_fp / n:.1f}':>36s}"
        print(row)

    print(f"\nNOISE, part 2: small real defects (found in how many of {n_patterns} noise patterns)")
    small_methods = {
        "gaussian": dict(),
        "median, min 20": dict(denoise="median"),
        "median, min 8": dict(denoise="median", min_defect_size=8),
        "median+gauss, min 8": dict(denoise="median+gaussian", min_defect_size=8),
    }
    for density in [0.0, 0.10]:
        print(f"  noise density {density:.0%}")
        print(f"  {'defect':22s}" + "".join(f"{name:>22s}" for name in small_methods))
        for label, (draw, truth_box) in SMALL_DEFECTS.items():
            row = f"  {label:22s}"
            for kwargs in small_methods.values():
                inspector = WireDefectInspector(ref, **kwargs)
                hits = 0
                for seed in seeds:
                    test = ref.copy()
                    draw(test)
                    if density:
                        test = add_salt_and_pepper(test, density, seed)
                    hits += score_detections(inspector.inspect(test)["defects"], truth=[truth_box])[0]
                row += f"{f'{hits}/{n_patterns}':>22s}"
            print(row)

    print(f"\nNOISE, part 3: 10% noise AND part shifted (4, -3) px + rotated 2 deg, aligned with ECC")
    combos = {
        "gaussian": dict(align="ecc"),
        "median 3x3 + gaussian": dict(align="ecc", denoise="median+gaussian", min_defect_size=8),
    }
    for name, kwargs in combos.items():
        inspector = WireDefectInspector(ref, **kwargs)
        clean_fp = found = defect_fp = 0
        for seed in seeds:
            clean_img = add_salt_and_pepper(misalign(clean_base, 4, -3, 2), 0.10, seed)
            defect_img = add_salt_and_pepper(misalign(defect_base, 4, -3, 2), 0.10, seed)
            clean_fp += len(inspector.inspect(clean_img)["defects"])
            f, fp = score_detections(inspector.inspect(defect_img)["defects"])
            found += f
            defect_fp += fp
        n = n_patterns
        print(f"  {name:24s} clean: {clean_fp / n:5.1f} false pos | defective: {found / n:.1f}/3 found, {defect_fp / n:5.1f} false pos")

    # picture: one noise pattern at 10%
    noisy_clean = add_salt_and_pepper(clean_base, 0.10, 0)
    noisy_defect = add_salt_and_pepper(defect_base, 0.10, 0)
    fig, axs = plt.subplots(2, len(methods) + 1, figsize=(4 * (len(methods) + 1), 5.2))
    show(axs[0][0], noisy_clean, "input: clean part + 10% noise")
    show(axs[1][0], noisy_defect, "input: defective part + 10% noise")
    for col, (name, kwargs) in enumerate(methods.items(), start=1):
        inspector = WireDefectInspector(ref, **kwargs)
        c = inspector.inspect(noisy_clean)
        d = inspector.inspect(noisy_defect)
        found, defect_fp = score_detections(d["defects"])
        show(axs[0][col], c["defect_img"], f"{name}\nclean: {len(c['defects'])} false positive(s)")
        show(axs[1][col], d["defect_img"], f"defective: {found}/3 found, {defect_fp} false pos.")
    fig.tight_layout()
    fig.savefig("demo_noise.png", dpi=100)


def main():
    ref = make_reference()
    basic_demo(ref)
    lighting_demo(ref)
    alignment_demo(ref)
    noise_demo(ref)
    plt.show()


if __name__ == "__main__":
    main()
