import cv2
import numpy as np


class HaarCascadeBase():

    def printVideoMessage(self, message='', key_message=''):
        if message == '':
            print('Starting Video Feed...')
            print('Press ESC to quit')
        else:
            print(message)
            print(key_message)

    def loadCascadeFile(self, cascade_file):

        if type(cascade_file) is list:
            cascades = []
            for cascade in cascade_file:
                cas = cv2.CascadeClassifier(cascade)
                cascades.append(cas)
            return cascades
        elif type(cascade_file) is str:
            cas = cv2.CascadeClassifier(cascade_file)
            return cas

    def displayEyesAndFaces(self, cascade_files, videoSource=0):
        cap = cv2.VideoCapture(videoSource)
        self.printVideoMessage()

        cascades = self.loadCascadeFile(cascade_files)
        face_cascade = cascades[0]
        eye_cascade = cascades[1]

        while True:
            _, frame = cap.read()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Original Video Feed', frame)

            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                eye_region_gray = gray_frame[y: y + h, x:x + w]
                eye_region_color = frame[y: y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(eye_region_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(eye_region_color, (ex, ey),
                                  (ex + ew, eh + eh), (0, 255, 0), 2)

            cv2.imshow('Faces and Eyes', frame)

            if cv2.waitKey(27) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
