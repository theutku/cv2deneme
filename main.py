import cv2
import numpy as np
import matplotlib.pyplot as plt

from basedisplay import BaseDisplay

if __name__ == '__main__':
    displayer = BaseDisplay()

    displayer.displayCvImage('sampleimg.jpg')
    displayer.displayCvImage('sampleimg.jpg', color=False)

    displayer.displayPlotImage('sampleimg.jpg')
    displayer.displayPlotImage('sampleimg.jpg', color=False)

    displayer.displayVideoFeed()
