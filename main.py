import cv2
import numpy as np
import matplotlib.pyplot as plt

from displaybase import BaseDisplay

if __name__ == '__main__':
    displayer = BaseDisplay()

    displayer.displayCvImage('sampleimg.jpg', legend=True)
    displayer.displayCvImage('sampleimg.jpg', color=False, legend=False)

    displayer.displayPlotImage('sampleimg.jpg')
    displayer.displayPlotImage('sampleimg.jpg', color=False)

    # displayer.displayVideoFeed(save=False)

# img = cv2.imread('sampleimg.jpg', cv2.IMREAD_COLOR)
# # cv2.imshow('image', img)
# # cv2.waitKey(0)

# # part = img[140:307, 294:370]
# # cv2.imshow('cut', part)
# # cv2.waitKey(0)

# # cv2.destroyAllWindows()

# plt.imshow(img, interpolation='bicubic')
# plt.show()
