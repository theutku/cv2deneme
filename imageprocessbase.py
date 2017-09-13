import numpy as np
import cv2


class ImageProcessorBase():

    def returnThreshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

        return ret, mask
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
