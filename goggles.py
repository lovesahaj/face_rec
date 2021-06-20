import face_recognition
import os
import cv2
from PIL import Image
import numpy as np

s_img = cv2.imread("goggles.png")
# print(s_img.shape)
# exit()

TOLERANCE = 0.45
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # hog or cnn

vid = cv2.VideoCapture(0)

print("processing faces")


while True:
    success, image = vid.read()
    location = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, location)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(f', found {len(encodings)} face(s)')

    alpha_s = 0.6
    alpha_l = 1.0 - alpha_s

    if len(encodings) > 0:
        for face_encoding, face_location in zip(encodings, location):
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            y1, y2 = face_location[3], face_location[1]
            x1, x2 = face_location[0] - 10, face_location[2] - 10

        s_img = cv2.resize(s_img, (y2 - y1, x2 - x1))

        imageGray = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
        _, imgBinary = cv2.threshold(imageGray, 50, 255, cv2.THRESH_BINARY)
        # imgBinary = cv2.adaptiveThreshold(imageGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                                   cv2.THRESH_BINARY, 199, 5)
        # imgBinary = imgBinary2 - imgBinary1
        imgBinary = cv2.cvtColor(imgBinary, cv2.COLOR_GRAY2RGB)

        # Inscribing the black region of imgBinary to main image(img) using bitwise_and operations
        image[x1:x2, y1:y2, :] = cv2.bitwise_and(
            image[x1:x2, y1:y2, :], imgBinary)

    # Adding the original color to the inscribed region using bitwise_or operations
    # image[x1:x2, y1:y2, :] = cv2.bitwise_or(image[x1:x2, y1:y2, :], imgBinary)

    # image[x1:x2, y1:y2, :] = cv2.addWeighted(
    #     s_img, alpha_s,
    #     image[x1:x2, y1:y2, :], alpha_l, 0
    # )

    cv2.imshow("Capture", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    os.system("cls")

vid.release()
cv2.destroyAllWindows()
