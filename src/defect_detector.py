import cv2
import numpy as np

class WireDefectInspector:
    def __init__(self, refImg, threshold=30, min_defect_size=20, contour_color=(0, 255, 0), contour_thickness=2, blur_kernel=(5, 5), morph_kernel=(3, 3),
                 threshold_mode="fixed", use_clahe=False, normalize_illumination=False, illum_ksize=101,
                 align=None, align_search=20, lock_x=False, denoise="gaussian", median_ksize=3):
        if threshold_mode not in ("fixed", "otsu", "adaptive"):
            raise ValueError(f"threshold_mode must be 'fixed', 'otsu' or 'adaptive', got {threshold_mode!r}")
        if align not in (None, "template", "ecc"):
            raise ValueError(f"align must be None, 'template' or 'ecc', got {align!r}")
        if denoise not in ("gaussian", "median", "median+gaussian"):
            raise ValueError(f"denoise must be 'gaussian', 'median' or 'median+gaussian', got {denoise!r}")
        self.denoise = denoise
        self.median_ksize = median_ksize
        # The reference gets the same median filter as every test image, so the filter's
        # side effects (it rounds corners and erases 1px features) cancel out in the diff.
        self.refImg = self._median(refImg)
        self.threshold = threshold
        self.min_defect_size = min_defect_size
        self.contour_color = contour_color
        self.contour_thickness = contour_thickness
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.threshold_mode = threshold_mode
        self.normalize_illumination = normalize_illumination
        self.illum_ksize = illum_ksize
        self.align = align
        self.align_search = align_search
        # For stock that looks the same at every x (wire, tape), nothing pins down the x position, so the
        # alignment drifts sideways and shifts the reported defect positions. lock_x=True only corrects y and rotation.
        self.lock_x = lock_x
        self.aligned = None  # did the last inspect() find an alignment? (None if align is off)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        
        gray_ref = cv2.cvtColor(self.refImg, cv2.COLOR_BGR2GRAY)
        self.ref_gray = gray_ref
        self.ref_illum = self._illumination(gray_ref)
        self.ref_blurred = self._gray_to_blurred(gray_ref, match_illumination=False)

    def inspect(self, testImg):
        # Median first: alignment resamples the image, which would smear each noise speck
        # into a bigger blob that the median can no longer remove. Note that the returned
        # defect_img is drawn on this denoised (and aligned) image, not on the raw input.
        testImg = self._median(testImg)
        testImg = self._align_to_reference(testImg)  # no-op unless align is set
        test_blurred = self._preprocess(testImg)
        thresh = self._compute_diff(test_blurred)    
        contours= self._find_defects(thresh)
        signed_diff = test_blurred.astype(np.int16) - self.ref_blurred  # >0: test brighter than reference
        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_defect_size:
                defects.append(self._describe(contour, area, signed_diff))

        defect_img = self.return_defect_image(contours, testImg)

        decision = "Defective" if defects else "Non-Defective"
        return {
            "status": decision,
            "defects": defects,
            "defect_img": defect_img,
            "aligned": self.aligned
        }
    
    def _describe(self, contour, area, signed_diff):
        x, y, w, h = cv2.boundingRect(contour)
        perimeter = cv2.arcLength(contour, True)
        (_, _), (side_a, side_b), _ = cv2.minAreaRect(contour)  # rotated box, so a diagonal line still reads as long and thin
        # mean brightness change over the defect: positive = extra material/brighter, negative = missing/darker
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1, offset=(-x, -y))
        return {
            "area": area, "x": x, "y": y, "w": w, "h": h,
            "circularity": 4 * np.pi * area / perimeter ** 2 if perimeter > 0 else 0.0,  # 1.0 = perfect circle
            "aspect_ratio": max(side_a, side_b) / max(min(side_a, side_b), 1.0),
            "polarity": float(signed_diff[y:y + h, x:x + w][mask > 0].mean()),
        }

    def _median(self, img):
        if self.denoise == "gaussian":
            return img
        return cv2.medianBlur(img, self.median_ksize)

    def _preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._gray_to_blurred(gray, match_illumination=True)

    def _align_to_reference(self, img):
        # Warps the whole BGR image, so the returned defects and the drawn outlines are in
        # the reference's coordinates and line up with the picture we return.
        if self.align is None:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        self.aligned = True
        if self.align == "template":
            # translation only, whole pixels: find where the reference (minus a border of
            # align_search px) sits inside the test image. Handles shifts up to align_search.
            pad = self.align_search
            template = self.ref_gray[pad:-pad, pad:-pad]
            scores = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, (x, y) = cv2.minMaxLoc(scores)
            warp = np.float32([[1, 0, pad - x], [0, 1, pad - y]])
            if self.lock_x:
                warp[0, 2] = 0
            return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # "ecc": translation + rotation, subpixel. Iterative, so it needs a starting point
        # that is already close (a few px, a few degrees).
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        try:
            _, warp = cv2.findTransformECC(self.ref_gray, gray, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
        except cv2.error:
            # No convergence: leave the image as is. The diff will then light up, so the
            # part fails safe as Defective; self.aligned tells you why.
            self.aligned = False
            return img
        if self.lock_x:
            warp[0, 2] = 0
        return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)

    def _illumination(self, gray):
        # Low-frequency brightness map. A median (not a Gaussian) so that a big defect
        # doesn't drag the estimate up and distort the gain around itself. Computed at
        # 1/4 size for speed. +1 avoids dividing by zero on black areas.
        h, w = gray.shape
        small = cv2.resize(gray, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        med = cv2.medianBlur(small, self.illum_ksize // 4 | 1)
        med = cv2.GaussianBlur(med, (0, 0), 4)
        return cv2.resize(med, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) + 1

    def _gray_to_blurred(self, gray, match_illumination):
        if match_illumination and self.normalize_illumination:
            # rescale the test image so its local brightness matches the reference
            gain = self.ref_illum / self._illumination(gray)
            gray = np.clip(gray * gain, 0, 255).astype(np.uint8)
        if self.clahe is not None:
            gray = self.clahe.apply(gray)
        if self.denoise == "median":
            return gray  # the median filter already smoothed it
        return cv2.GaussianBlur(gray, self.blur_kernel, 0)
    
    def _compute_diff(self, test_blurred):
        diff = cv2.absdiff(self.ref_blurred, test_blurred)
        thresh = self._threshold(diff)

        kernel = np.ones(self.morph_kernel, np.uint8)
        solidMask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleanMask = cv2.morphologyEx(solidMask, cv2.MORPH_CLOSE, kernel)
        return cleanMask
    
    def _threshold(self, diff):
        if self.threshold_mode == "otsu":
            return cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if self.threshold_mode == "adaptive":
            # pixel must exceed its 31x31 local mean by self.threshold
            return cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, -self.threshold)
        return cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)[1]

    def _find_defects(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def return_defect_image(self, contours, testImg):
        defect_img = testImg.copy()
        for contour in contours:
            if cv2.contourArea(contour) > self.min_defect_size:
                cv2.drawContours(defect_img, [contour], -1, self.contour_color, self.contour_thickness)
        return defect_img
    