#!/usr/bin/python3
import cv2 as cv
import os

assets = os.environ['MDS_ASSETS']

def loadImage(image_name):
    image_path = assets + "/Images/" + image_name
    image = cv.imread(image_path, cv.IMREAD_COLOR)

    if image is None:
        raise Exception("[ERROR] Image not found.")

    scaled_height = int(image.shape[0] * 0.60)
    scaled_width = int(image.shape[1] * 0.60)

    image = cv.resize(image, (scaled_width, scaled_height), interpolation=cv.INTER_AREA)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image, image_gray

# Detect images using a pre-trained classifier
def detect_frontfaces(image_color, image_gray):
    face_detector = cv.CascadeClassifier(assets + "/Cascades/haarcascade_frontalface_default.xml")
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.09, minNeighbors=5)
    for (x, y, rec_width, rec_height) in detections:
        cv.rectangle(image_color, (x, y), (x + rec_width,y + rec_height), (0, 255, 0), 3)
    print(detections)

def detect_eyes(image_color, image_gray):
    eye_detector = cv.CascadeClassifier(assets + "/Cascades/haarcascade_eye.xml")
    detections = eye_detector.detectMultiScale(image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10), maxSize=(30, 30))
    for (x, y, rec_width, rec_height) in detections:
        cv.rectangle(image_color, (x, y), (x + rec_width,y + rec_height), (0, 0, 255), 2)
    print(detections)

def main():
    try:
        images_names = ["people1.jpg", "people2.jpg"]
        for image_name in images_names:
            image_color, image_gray = loadImage(image_name)
            detect_frontfaces(image_color, image_gray)
            detect_eyes(image_color, image_gray)
            print(image_color.shape)
            cv.imshow("Detections", image_color)
            cv.waitKey(10000)
    except Exception as e:
        print("Exception:", str(e))

if __name__ == "__main__":
    main()
