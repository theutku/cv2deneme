import cv2
import numpy as np
import matplotlib.pyplot as plt

from imagetextbase import ImageTextInput
from imageprocessbase import ImageProcessorBase


class BaseDisplay():

    def displayAndClose(self, image, title='', closeWhenfinished=True):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        if closeWhenfinished is True:
            cv2.destroyAllWindows()

    def displayCvImage(self, imageName, color=True, save=False, legend=True):
        colorOption = cv2.IMREAD_COLOR if color is True else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(imageName, colorOption)

        if legend is True:
            ImageTextInput().addText(img, 'Press any key to continue...')

        if color is False & save is True:
            cv2.imwrite('savedimage.png', img)

        self.displayAndClose(img, 'Image')

    def displayPlotImage(self, imageName, color=True):
        colorOption = cv2.IMREAD_COLOR if color is True else cv2.IMREAD_GRAYSCALE
        img = cv2.imread(imageName, colorOption)

        plt.imshow(img, cmap='gray', interpolation='bicubic')
        plt.show()

    def displayVideoFeed(self, videoSource=0, grayscale=False, save=False):
        print('Starting Video Feed...')
        print('Press Q to quit')
        cap = cv2.VideoCapture(videoSource)

        if save is True:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('savedvideo.avi', fourcc, 20.0, (640, 480))

        while True:
            ret, frame = cap.read()
            if grayscale is True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow('Gray', gray)

            if save is True:
                out.write(frame)

            cv2.imshow('Video Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if save is True:
            out.release()
        cv2.destroyAllWindows()

    def displayThreshold(self, image):
        ret, mask = ImageProcessorBase().returnThreshold(image)
        self.displayAndClose(mask, 'Threshold')
