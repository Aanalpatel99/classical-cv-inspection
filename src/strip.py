"""Simulated line-scan wire inspection: a long thin strip, inspected chunk by chunk."""
from collections import defaultdict

import cv2
import numpy as np

from src.defect_detector import WireDefectInspector
from src.utils import add_salt_and_pepper, box_iou

STRIP_H = 100
WIRE_TOP, WIRE_BOTTOM = 30, 70  # the wire fills rows 30..69 (40 px thick); everything else is background
WIRE_GRAY = 128
CHUNK_W = 500
OVERLAP = 60  # must be wider than the widest defect, or a defect cut by a chunk edge is never seen whole
EDGE_MARGIN = 2  # a detection this close to a chunk's edge may be cut off
EDGE_BAND = 4  # rows within this distance of a wire edge count as "on the edge"
STRADDLE = 4  # a defect reaching this far past a wire edge on BOTH sides crosses it
DEFECT_TYPES = ("pinhole", "scratch", "lump", "neckdown")


def make_wire(width=CHUNK_W):
    """A defect-free stretch of wire. Uniform along its length, so one of these is the reference for every chunk."""
    gray = np.zeros((STRIP_H, width), np.uint8)
    gray[WIRE_TOP:WIRE_BOTTOM, :] = WIRE_GRAY
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _draw(img, kind, x, rng, varied=False, edge_scratch=False):
    """Draw one defect of `kind` centred at column x. Modifies img in place.
    varied=False: every defect of a type looks the same. varied=True: random sizes, brightness and angles."""
    if kind == "pinhole":  # small dark round spot inside the wire
        cy = int(rng.integers(WIRE_TOP + 12, WIRE_BOTTOM - 12))
        radius, level = (int(rng.integers(3, 7)), int(rng.integers(0, 90))) if varied else (4, 30)
        cv2.circle(img, (x, cy), radius, (level,) * 3, -1)
    elif kind == "scratch":  # thin line, normally well inside the wire
        if varied:
            length = rng.uniform(16, 40)
            level = int(rng.integers(170, 241)) if rng.random() < 0.7 else int(rng.integers(20, 91))  # bright or dark
            thickness = int(rng.integers(1, 4))
            if edge_scratch:  # crosses a wire edge at a slant
                cy = int(rng.choice([WIRE_TOP, WIRE_BOTTOM]))
                angle = np.radians(rng.choice([-1, 1]) * rng.uniform(25, 70))
            else:  # steep scratches must stay clear of the edges, so the allowed angle shrinks as they get longer
                cy = 50
                max_angle = min(80.0, np.degrees(np.arcsin(min(1.0, 26 / length))))
                angle = np.radians(rng.uniform(-max_angle, max_angle))
            half = length / 2
        else:
            angle = np.radians(rng.uniform(-45, 45))
            half, level, thickness, cy = 12, 200, 2, 50
        dx, dy = half * np.cos(angle), half * np.sin(angle)
        cv2.line(img, (round(x - dx), round(cy - dy)), (round(x + dx), round(cy + dy)), (level,) * 3, thickness)
    elif kind in ("lump", "neckdown"):
        edge = int(rng.choice([WIRE_TOP, WIRE_BOTTOM]))  # top or bottom edge
        if kind == "lump":  # extra material bulging out past the edge
            axes = (int(rng.integers(7, 15)), int(rng.integers(5, 11))) if varied else (10, 8)
            cv2.ellipse(img, (x, edge), axes, 0, 0, 360, (WIRE_GRAY,) * 3, -1)
        else:  # material missing: the wire narrows
            axes = (int(rng.integers(8, 17)), int(rng.integers(4, 10))) if varied else (12, 7)
            cv2.ellipse(img, (x, edge), axes, 0, 0, 360, (0, 0, 0), -1)
    else:
        raise ValueError(kind)


def wander_offset(seed, length, amplitude):
    """Vertical shift in px of every column of a wandering wire (what make_strip applies when wander=amplitude)."""
    wrng = np.random.default_rng([seed, 1])  # separate stream, so turning wander on does not change the defects
    period, phase = wrng.uniform(3000, 6000), wrng.uniform(0, 2 * np.pi)
    return amplitude * np.sin(2 * np.pi * np.arange(length) / period + phase)


def make_strip(length=6000, n_defects=12, seed=0, noise_sigma=0.0, varied=False, wander=0.0, edge_scratches=False,
               speck_density=0.0):
    """A long wire with random defects. Returns (strip, truth), where truth is a list of
    {"type": ..., "box": (x, y, w, h)} for every defect that was drawn.

    varied:         random defect sizes/brightness/angles instead of one fixed look per type
    wander:         amplitude in px of a slow up-and-down drift of the whole wire (0 = perfectly straight).
                    Truth boxes stay in the straight, undrifted frame, which is where an aligned inspector reports.
    edge_scratches: some scratches cross a wire edge (hard to tell from a lump or neckdown); needs varied=True
    noise_sigma:    Gaussian sensor noise added to every pixel
    speck_density:  fraction of pixels turned pure black or white (salt-and-pepper noise)"""
    rng = np.random.default_rng(seed)
    strip = make_wire(length)
    xs = np.linspace(200, length - 200, n_defects) + rng.integers(-60, 60, n_defects)
    truth = []
    for x in xs.astype(int):
        kind = str(rng.choice(DEFECT_TYPES))
        edge_scratch = bool(edge_scratches and kind == "scratch" and rng.random() < 0.5)
        before = strip.copy()
        _draw(strip, kind, int(x), rng, varied, edge_scratch)
        changed = np.any(strip != before, axis=2).astype(np.uint8)
        truth.append({"type": kind, "box": cv2.boundingRect(changed), "on_edge": edge_scratch})  # exact box of the pixels that changed
    if wander:
        offset = wander_offset(seed, length, wander)
        map_x = np.tile(np.arange(length, dtype=np.float32), (STRIP_H, 1))
        map_y = np.arange(STRIP_H, dtype=np.float32)[:, None] - offset.astype(np.float32)[None, :]
        strip = cv2.remap(strip, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if noise_sigma:
        strip = np.clip(strip + rng.normal(0, noise_sigma, strip.shape), 0, 255).astype(np.uint8)
    if speck_density:
        strip = add_salt_and_pepper(strip, speck_density, seed)
    return strip, truth


def classify_defect(d):
    """Name a detected defect from its shape, polarity and position (see WireDefectInspector._describe)."""
    top, bottom = d["y"], d["y"] + d["h"]
    for edge in (WIRE_TOP, WIRE_BOTTOM):
        if top <= edge - STRADDLE and bottom >= edge + STRADDLE:
            return "scratch"  # reaches both outside and inside the wire, so it crosses the edge. A lump is only outside, a neckdown only inside
    if any(top < edge + EDGE_BAND and bottom > edge - EDGE_BAND for edge in (WIRE_TOP, WIRE_BOTTOM)):
        return "lump" if d["polarity"] > 0 else "neckdown"  # extra material vs missing material
    if d["aspect_ratio"] >= 3:
        return "scratch"
    if d["polarity"] < 0:
        return "pinhole"
    return "unknown"


def inspect_strip(strip, inspector, chunk_w=CHUNK_W, overlap=OVERLAP, stats=None):
    """Inspect a long strip in overlapping chunks. The inspector's reference must be make_wire(chunk_w).
    Returns detections in strip coordinates, each with a "type". If a dict is passed as `stats`,
    stats["failed_alignments"] is incremented for each chunk the inspector could not align."""
    height, width = strip.shape[:2]
    starts = list(range(0, width - chunk_w + 1, chunk_w - overlap))
    if starts[-1] != width - chunk_w:
        starts.append(width - chunk_w)  # last chunk sits flush against the end

    found = []
    for x0 in starts:
        result = inspector.inspect(strip[:, x0:x0 + chunk_w])
        if stats is not None and result["aligned"] is False:
            stats["failed_alignments"] = stats.get("failed_alignments", 0) + 1
        for d in result["defects"]:
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


def evaluate_strips(n_strips, settings=None, **strip_kwargs):
    """Run the whole pipeline on n_strips random strips (seeds 0..n_strips-1).
    `settings` go to WireDefectInspector, `strip_kwargs` to make_strip.
    Returns a dict: total, missed, false_pos, wrong, failed_alignments, confusion {(true, predicted): count},
    features {true type: [[area, circularity, aspect_ratio, polarity], ...]}."""
    inspector = WireDefectInspector(make_wire(), **(settings or {}))
    out = {"total": 0, "missed": 0, "false_pos": 0, "confusion": defaultdict(int), "features": defaultdict(list)}
    stats = {}
    for seed in range(n_strips):
        strip, truth = make_strip(seed=seed, **strip_kwargs)
        pairs, extra = match_detections(inspect_strip(strip, inspector, stats=stats), truth)
        out["total"] += len(truth)
        out["false_pos"] += len(extra)
        for t, d in pairs:
            if d is None:
                out["missed"] += 1
                continue
            out["confusion"][(t["type"], d["type"])] += 1
            out["features"][t["type"]].append([d["area"], d["circularity"], d["aspect_ratio"], d["polarity"]])
    out["failed_alignments"] = stats.get("failed_alignments", 0)
    out["wrong"] = sum(n for (true, pred), n in out["confusion"].items() if true != pred)
    return out
