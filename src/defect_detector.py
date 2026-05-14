import cv2
import numpy as np

class WireDefectInspector:
    def __init__(self, refImg, threshold=30, min_defect_size=100, contour_color=(0, 255, 0), contour_thickness=2, kernel_size=(5, 5)):
        self.refImg = refImg
        self.threshold = threshold
        self.min_defect_size = min_defect_size
        self.contour_color = contour_color
        self.contour_thickness = contour_thickness
        self.kernel_size = kernel_size
        
        gray_ref = cv2.cvtColor(self.refImg, cv2.COLOR_BGR2GRAY)
        self.ref_blurred = cv2.GaussianBlur(gray_ref, self.kernel_size, 0)

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
        blurred = cv2.GaussianBlur(gray, self.kernel_size, 0)
        return blurred
    
    def _compute_diff(self, test_blurred):
        diff = cv2.absdiff(self.ref_blurred, test_blurred)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones(self.kernel_size, np.uint8)
        solidMask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleanMask = cv2.morphologyEx(solidMask, cv2.MORPH_CLOSE, kernel)
        return cleanMask
    
    def _find_defects(self, thresh):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def return_defect_image(self, contours, testImg):
        defect_img = testImg.copy()
        for contour in contours:
            if cv2.contourArea(contour) > self.min_defect_size:
                cv2.drawContours(defect_img, [contour], -1, self.contour_color, self.contour_thickness)
        return defect_img
    