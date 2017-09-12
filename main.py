import cv2
import numpy as np
import matplotlib.pyplot as plt


# img = cv2.imread('sampleimg.jpg', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('sampleimg.jpg', cv2.IMREAD_COLOR)
# img = cv2.imread('sampleimg.jpg', cv2.IMREAD_UNCHANGED)

# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# plt.imshow(img, cmap='gray', interpolation='bicubic')
# plt.plot([50, 100], [80, 100], 'c', linewidth=4)
# plt.show()

# cv2.imwrite('sampleimggray.png', img)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
