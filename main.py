import cv2
import numpy as np
import matplotlib.pyplot as plt

from opencv.displaybase import BaseDisplay
from opencv.cascade.cascadebase import HaarCascadeBase
from opencv.cascade.downloadbase import CascadeImageProcessor


def get_download_from_user():
    download_pics = {
        'y': 'Yes',
        'n': 'No'
    }

    prompt = ''
    for key, value in download_pics.items():
        line = 'Type {0} for {1}\n'.format(key, value)
        prompt += line

    user_selection = input(
        'Download Haar Cascade pictures?\n{}'.format(prompt))

    return download_pics.get(user_selection, 'n')


if __name__ == '__main__':
    displayer = BaseDisplay()

    cascadeBase = HaarCascadeBase()

    cas = CascadeImageProcessor('downloads')

    download_preference = get_download_from_user()
    if download_preference == 'Yes':
        cas.prepare_store_images()

    cas.remove_uglies()
    cas.create_desc_files()
    # displayer.displayCvImage('img/sampleimg.jpg', legend=True)
    # displayer.displayCvImage('img/sampleimg.jpg', color=False, legend=False)

    # displayer.displayPlotImage('img/sampleimg.jpg')
    # displayer.displayPlotImage('img/sampleimg.jpg', color=False)

    # displayer.displayThreshold('img/python.png')

    # displayer.displayVideoFeed(grayscale=True, save=False)

    # # Show only red colors
    # displayer.displayFilteredCam(videoSource=0, lower_color=[
    #     150, 150, 50], upper_color=[180, 255, 150], morphology='all')

    # displayer.displayGradients(videoSource=0, method='all')

    # displayer.detectEdges(videoSource=0, row_size=100, column_size=200)

    # displayer.matchTemplate('img/templateimg/sampleimage.jpg',
    #                         'img/templateimg/templateimage.jpg')

    # displayer.displayForeground(
    #     'img/foreground/foreground-image.jpg', rectangle=(50, 50, 300, 500))

    # displayer.displayCorners('img/corners/corner-sample.jpg',
    # feature_count=100, quality=0.1, minimum_distance=10)

    # displayer.displayFeatureMatch(
    #     'img/feature_match/main-image.jpg', 'img/feature_match/feature.jpg', feature_number=10)

    # displayer.displayMotionReduction(
    #     videoSource='img/background_reduction/people-walking.mp4',
    #     noise_reduction='gaussian')

    # cascadeBase.displayEyesAndFaces(
    #     cascade_files=['data/haarcascades/haarcascade_frontalface_default.xml',
    #                    'data/haarcascades/haarcascade_eye.xml'], videoSource=0)
