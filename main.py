import cv2
import numpy as np
import matplotlib.pyplot as plt

from opencv.displaybase import BaseDisplay
from opencv.cascade.cascadebase import HaarCascadeBase
from opencv.cascade.downloadbase import DownloadPath


displayer = BaseDisplay()
cascadeBase = HaarCascadeBase('downloads')

if __name__ == '__main__':

    download_preference = DownloadPath.get_user_request(
        'Download Haar Cascade pictures?')
    if download_preference == 'Yes':
        cascadeBase.prepare_store_images()

    cascadeBase.create_desc_files()
    cascadeBase.remove_uglies()

    generate_positives = DownloadPath.get_user_request(
        'Generate Positive Images?')
    if generate_positives == 'Yes':
        cascadeBase.form_positive_images(
            file_name='info', maxxangle=0.5, maxyangle=-0.5, maxzangle=0.5)
        cascadeBase.form_positive_vector(
            file_name='positives', width=20, height=20)

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

    # cascadeBase.display_profile_face(
    #     'data/haarcascades/haarcascade_profileface.xml', videoSource=0)
