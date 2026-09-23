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
