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

    displayer.displayThreshold('python.png')

    displayer.displayVideoFeed(grayscale=True, save=False)

    # Show only red colors
    displayer.displayFilteredCam(videoSource=0, lower_color=[
        150, 150, 50], upper_color=[180, 255, 150], morphology='all')

    displayer.displayGradients(videoSource=0, method='all')

    displayer.detectEdges(videoSource=0, row_size=100, column_size=200)

    displayer.matchTemplate('templateimg/sampleimage.jpg',
                            'templateimg/templateimage.jpg')
