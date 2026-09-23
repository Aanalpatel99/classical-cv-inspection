"""Simulated line-scan wire inspection: a long thin strip, inspected chunk by chunk."""
import cv2
import numpy as np

from src.utils import box_iou

STRIP_H = 100
WIRE_TOP, WIRE_BOTTOM = 30, 70  # the wire fills rows 30..69 (40 px thick); everything else is background
WIRE_GRAY = 128
CHUNK_W = 500
OVERLAP = 60  # must be wider than the widest defect, or a defect cut by a chunk edge is never seen whole
EDGE_MARGIN = 2  # a detection this close to a chunk's edge may be cut off
EDGE_BAND = 4  # rows within this distance of a wire edge count as "on the edge"
DEFECT_TYPES = ("pinhole", "scratch", "lump", "neckdown")


def make_wire(width=CHUNK_W):
    """A defect-free stretch of wire. Uniform along its length, so one of these is the reference for every chunk."""
    gray = np.zeros((STRIP_H, width), np.uint8)
    gray[WIRE_TOP:WIRE_BOTTOM, :] = WIRE_GRAY
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw(img, kind, x, rng):
    """Draw one defect of `kind` centred at column x. Modifies img in place."""
    if kind == "pinhole":  # small dark round spot inside the wire
        cv2.circle(img, (x, int(rng.integers(WIRE_TOP + 12, WIRE_BOTTOM - 12))), 4, (30, 30, 30), -1)
    elif kind == "scratch":  # thin bright line at a random angle, well inside the wire
        angle = np.radians(rng.uniform(-45, 45))
        dx, dy = 12 * np.cos(angle), 12 * np.sin(angle)
        cv2.line(img, (round(x - dx), round(50 - dy)), (round(x + dx), round(50 + dy)), (200, 200, 200), 2)
    elif kind in ("lump", "neckdown"):
        edge = int(rng.choice([WIRE_TOP, WIRE_BOTTOM]))  # top or bottom edge
        if kind == "lump":  # extra material bulging out past the edge
            cv2.ellipse(img, (x, edge), (10, 8), 0, 0, 360, (WIRE_GRAY,) * 3, -1)
        else:  # material missing: the wire narrows
            cv2.ellipse(img, (x, edge), (12, 7), 0, 0, 360, (0, 0, 0), -1)
    else:
        raise ValueError(kind)


def make_strip(length=6000, n_defects=12, seed=0, noise_sigma=0.0):
    """A long wire with random defects. Returns (strip, truth), where truth is a list of
    {"type": ..., "box": (x, y, w, h)} for every defect that was drawn."""
    rng = np.random.default_rng(seed)
    strip = make_wire(length)
    xs = np.linspace(200, length - 200, n_defects) + rng.integers(-60, 60, n_defects)
    truth = []
    for x in xs.astype(int):
        kind = str(rng.choice(DEFECT_TYPES))
        before = strip.copy()
        _draw(strip, kind, int(x), rng)
        changed = np.any(strip != before, axis=2).astype(np.uint8)
        truth.append({"type": kind, "box": cv2.boundingRect(changed)})  # exact box of the pixels that changed
    if noise_sigma:
        strip = np.clip(strip + rng.normal(0, noise_sigma, strip.shape), 0, 255).astype(np.uint8)
    return strip, truth


def classify_defect(d):
    """Name a detected defect from its shape and polarity (see WireDefectInspector._describe)."""
    on_edge = d["y"] < WIRE_TOP + EDGE_BAND and d["y"] + d["h"] > WIRE_TOP - EDGE_BAND \
        or d["y"] < WIRE_BOTTOM + EDGE_BAND and d["y"] + d["h"] > WIRE_BOTTOM - EDGE_BAND
    if on_edge:
        return "lump" if d["polarity"] > 0 else "neckdown"  # extra material vs missing material
    if d["aspect_ratio"] >= 3:
        return "scratch"
    if d["polarity"] < 0:
        return "pinhole"
    return "unknown"


def inspect_strip(strip, inspector, chunk_w=CHUNK_W, overlap=OVERLAP):
    """Inspect a long strip in overlapping chunks. The inspector's reference must be make_wire(chunk_w).
    Returns detections in strip coordinates, each with a "type"."""
    height, width = strip.shape[:2]
    starts = list(range(0, width - chunk_w + 1, chunk_w - overlap))
    if starts[-1] != width - chunk_w:
        starts.append(width - chunk_w)  # last chunk sits flush against the end

    found = []
    for x0 in starts:
        for d in inspector.inspect(strip[:, x0:x0 + chunk_w])["defects"]:
            cut_left = x0 > 0 and d["x"] <= EDGE_MARGIN
            cut_right = x0 + chunk_w < width and d["x"] + d["w"] >= chunk_w - EDGE_MARGIN
            if cut_left or cut_right:
                continue  # cut by a chunk edge; the overlapping neighbour chunk sees it whole
            det = dict(d, x=x0 + d["x"], type=classify_defect(d))
            if not any(box_iou((det["x"], det["y"], det["w"], det["h"]), (o["x"], o["y"], o["w"], o["h"])) > 0
                       for o in found):  # inside the overlap zone, so both chunks saw it
                found.append(det)
    return found


def match_detections(found, truth, min_iou=0.3):
    """Pair each true defect with its best-overlapping detection. Returns (pairs, false_positives),
    where pairs is [(true_defect, detection or None)] and false_positives are unmatched detections."""
    used = set()
    pairs = []
    for t in truth:
        best, best_iou = None, min_iou
        for i, d in enumerate(found):
            iou = box_iou(t["box"], (d["x"], d["y"], d["w"], d["h"]))
            if i not in used and iou >= best_iou:
                best, best_iou = i, iou
        if best is not None:
            used.add(best)
        pairs.append((t, found[best] if best is not None else None))
    return pairs, [d for i, d in enumerate(found) if i not in used]
