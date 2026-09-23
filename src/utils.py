import cv2
import numpy as np

IMG_H, IMG_W = 200, 600
PART_GRAY = 128


def make_reference():
    """Defect-free reference: a gray rectangle (the part) on a black background, as BGR."""
    gray = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    cv2.rectangle(gray, (50, 50), (550, 150), PART_GRAY, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_test(ref, big_defect=True, thin_line=True, short_line=True, noise_dots=True):
    """Copy of the reference with synthetic defects drawn on it.

    big_defect: 35px-radius white circle (should be caught)
    thin_line:  2x30px white line (small but real, should be caught)
    short_line: 2x20px white line (smaller still, should be caught)
    noise_dots: four 1px dots (sensor noise, should be ignored)
    """
    test = ref.copy()
    white = (255, 255, 255)
    if big_defect:
        cv2.circle(test, (100, 100), 35, white, -1)
    if thin_line:
        cv2.line(test, (400, 100), (400, 130), white, 2)
    if short_line:
        cv2.line(test, (500, 100), (520, 100), white, 2)
    if noise_dots:
        for center in [(70, 70), (180, 60), (500, 135), (150, 120)]:
            cv2.circle(test, center, 1, white, -1)
    return test


def apply_lighting_gradient(img, left=0.7, right=1.3):
    """Simulate uneven lighting: scale brightness linearly from `left` to `right` across the width."""
    gain = np.linspace(left, right, img.shape[1], dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def misalign(img, dx=0, dy=0, angle=0.0):
    """Simulate a part that arrived shifted (dx, dy px) and/or rotated (degrees) relative to the reference."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += (dx, dy)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# (x, y, w, h) of the three real defects that make_test() draws
TRUE_DEFECT_BOXES = [(65, 65, 71, 71), (398, 99, 5, 33), (499, 98, 23, 5)]


def score_detections(defects, truth=TRUE_DEFECT_BOXES, min_iou=0.3):
    """Return (real defects found, false positives). A detection is a hit if it overlaps a
    true defect box with IoU >= min_iou. Counting detections alone hides false positives."""
    def iou(a, b):
        ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / (a[2] * a[3] + b[2] * b[3] - inter)

    boxes = [(d["x"], d["y"], d["w"], d["h"]) for d in defects]
    found = sum(any(iou(t, b) >= min_iou for b in boxes) for t in truth)
    false_positives = sum(not any(iou(t, b) >= min_iou for t in truth) for b in boxes)
    return found, false_positives


def add_salt_and_pepper(img, density=0.05, seed=None):
    """Set a random `density` fraction of pixels to pure black (pepper) or pure white (salt), half each."""
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape[:2])
    out = img.copy()
    out[r < density / 2] = 0
    out[(r >= density / 2) & (r < density)] = 255
    return out


# Small real defects, each with the (x, y, w, h) box it should be detected in.
# Used to check whether a smoothing filter erases things we care about.
SMALL_DEFECTS = {
    "2px line, 30 long": (lambda t: cv2.line(t, (400, 100), (400, 130), (255, 255, 255), 2), (399, 99, 3, 32)),
    "1px line, 30 long": (lambda t: cv2.line(t, (400, 100), (400, 130), (255, 255, 255), 1), (399, 99, 2, 32)),
    "2px line, 10 long": (lambda t: cv2.line(t, (400, 100), (400, 110), (255, 255, 255), 2), (399, 99, 3, 12)),
    "dot, radius 3":     (lambda t: cv2.circle(t, (300, 100), 3, (255, 255, 255), -1), (296, 96, 9, 9)),
    "dot, radius 2":     (lambda t: cv2.circle(t, (300, 100), 2, (255, 255, 255), -1), (297, 97, 7, 7)),
}
