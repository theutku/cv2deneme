import cv2
import numpy as np
import matplotlib.pyplot as plt

from displaybase import BaseDisplay

if __name__ == '__main__':
    displayer = BaseDisplay()

    # displayer.displayCvImage('sampleimg.jpg', legend=True)
    # displayer.displayCvImage('sampleimg.jpg', color=False, legend=False)

    # displayer.displayPlotImage('sampleimg.jpg')
    # displayer.displayPlotImage('sampleimg.jpg', color=False)

    displayer.displayThreshold('python.png')

    # displayer.displayVideoFeed(grayscale=True, save=False)
