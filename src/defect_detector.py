import cv2
import numpy as np

class WireDefectInspector:
    def __init__(self, refImg, threshold=30, min_defect_size=20, contour_color=(0, 255, 0), contour_thickness=2, blur_kernel=(5, 5), morph_kernel=(3, 3),
                 threshold_mode="fixed", use_clahe=False, normalize_illumination=False, illum_ksize=101):
        if threshold_mode not in ("fixed", "otsu", "adaptive"):
            raise ValueError(f"threshold_mode must be 'fixed', 'otsu' or 'adaptive', got {threshold_mode!r}")
        self.refImg = refImg
        self.threshold = threshold
        self.min_defect_size = min_defect_size
        self.contour_color = contour_color
        self.contour_thickness = contour_thickness
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel
        self.threshold_mode = threshold_mode
        self.normalize_illumination = normalize_illumination
        self.illum_ksize = illum_ksize
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
        
        gray_ref = cv2.cvtColor(self.refImg, cv2.COLOR_BGR2GRAY)
        self.ref_illum = self._illumination(gray_ref)
        self.ref_blurred = self._gray_to_blurred(gray_ref, match_illumination=False)

    def inspect(self, testImg):
        test_blurred = self._preprocess(testImg)
        thresh = self._compute_diff(test_blurred)    
        contours= self._find_defects(thresh)
        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_defect_size:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append({"area": area, "x": x, "y": y, "w": w, "h": h})

        defect_img = self.return_defect_image(contours, testImg)

        decision = "Defective" if defects else "Non-Defective"
        return {
            "status": decision,
            "defects": defects,
            "defect_img": defect_img
        }
    
    def _preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._gray_to_blurred(gray, match_illumination=True)

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
    