import numpy as np
import cv2


class ImageProcessorBase():

    def returnThreshold(self, image):
        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        inv_mask = cv2.bitwise_not(mask)

        return ret, img, mask, inv_mask
