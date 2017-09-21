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

    # displayer.displayThreshold('python.png')

    # displayer.displayVideoFeed(grayscale=True, save=False)

    # # Show only red colors
    # displayer.displayFilteredCam(videoSource=0, lower_color=[
    #     150, 150, 50], upper_color=[180, 255, 150], morphology='all')

    # displayer.displayGradients(videoSource=0, method='all')

    # displayer.detectEdges(videoSource=0, row_size=100, column_size=200)

    # displayer.matchTemplate('templateimg/sampleimage.jpg',
    #                         'templateimg/templateimage.jpg')

    # displayer.displayForeground(
    #     'foreground/foreground-image.jpg', rectangle=(50, 50, 300, 500))

    # displayer.displayCorners('corners/corner-sample.jpg',
    # feature_count=100, quality=0.1, minimum_distance=10)

    # displayer.displayFeatureMatch(
    #     'feature_match/main-image.jpg', 'feature_match/feature.jpg', feature_number=10)

    displayer.displayReducedBackground(
        videoSource='background_reduction/people-walking.mp4')
