import cv2
import numpy as np
import matplotlib.pyplot as plt

from imagetextbase import ImageTextInput
from imageprocessbase import ImageProcessorBase


class BaseDisplay():

    def displayAndClose(self, image, title='', closeWhenFinished=True):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        if closeWhenFinished is True:
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
            _, frame = cap.read()
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
        ret, img, mask, inv_mask = ImageProcessorBase().returnThreshold(image)
        self.displayAndClose(img, 'Original Image', closeWhenFinished=False)
        self.displayAndClose(mask, 'Threshold', closeWhenFinished=False)
        self.displayAndClose(inv_mask, 'InverseThreshold')

    def displayFilteredCam(self, videoSource=0, lower_color=[0, 0, 0], upper_color=[255, 255, 255]):
        print('Starting Filtered Video Feed...')
        print('Press ESC to quit')

        cap = cv2.VideoCapture(videoSource)

        while True:
            _, frame = cap.read()

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red = np.array(lower_color)
            upper_red = np.array(upper_color)

            mask = cv2.inRange(hsv, lower_red, upper_red)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow('Original Video Output', frame)
            cv2.imshow('Mask', mask)
            cv2.imshow('Filtered Video Output', result)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
