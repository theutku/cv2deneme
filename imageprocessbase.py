import numpy as np
import cv2


class ImageProcessorBase():

    smoothings = {
        'kernel': 'Kernel Smoothing',
        'gaussian': 'Gaussian Blur Smoothing',
        'median': 'Median Blur Smoothing'
    }

    morphs = {
        'erosion': 'Erosion Morphology',
        'dilation': 'Dilation Morphology',
        'opening': 'Opening Morphology',
        'closing': 'Closing Morphology'
    }

    def returnThreshold(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        return ret, img, mask, inv_mask

    def addBasicSmoothing(self, frame, smoothing_method='median'):
        kernel = np.ones((15, 15), np.float) / 255
        smoothed = cv2.filter2D(frame, -1, kernel)
        gblur = cv2.GaussianBlur(frame, (15, 15), 0)
        median_blur = cv2.medianBlur(frame, 15)

        if smoothing_method == 'kernel':
            return smoothed, self.smoothings[smoothing_method]
        elif smoothing_method == 'gaussian':
            return gblur, self.smoothings[smoothing_method]
        elif smoothing_method == 'median':
            return median_blur, self.smoothings[smoothing_method]
        elif smoothing_method == 'all':
            cv2.imshow('Kernel Smoothing', smoothed)
            cv2.imshow('Gaussian Blur Smoothing', gblur)
            cv2.imshow('Median Blur Smoothing', median_blur)

    def addMorphology(self, frame, morph_medthod='closing'):
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(frame, kernel, iterations=1)
        dilation = cv2.dilate(frame, kernel, iterations=1)

        opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

        if morph_medthod == 'erosion':
            return erosion, self.morphs[morph_medthod]
        elif morph_medthod == 'dilation':
            return dilation, self.morphs[morph_medthod]
        elif morph_medthod == 'opening':
            return opening, self.morphs[morph_medthod]
        elif morph_medthod == 'closing':
            return closing, self.morphs[morph_medthod]
        elif morph_medthod == 'all':
            cv2.imshow('Erosion Morphology', erosion)
            cv2.imshow('Dilation Morphology', dilation)
            cv2.imshow('Opening Morphology', opening)
            cv2.imshow('Closing Morphology', closing)
