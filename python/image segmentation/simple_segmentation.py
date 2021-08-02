#!/usr/bin/python3
from skimage.color import rgb2gray
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy import ndimage
import os

img_assets = os.environ["MDS_ASSETS"] + "/Images/"

# Segmentation using the mean as a threshold
def black_and_white_segmentation(image_gray):
    img_gray_tmp = image_gray.reshape(image_gray.shape[0] * image_gray.shape[1])

    for i in range(img_gray_tmp.shape[0]):
        pixel_val = img_gray_tmp[i]
        img_gray_tmp[i] = 1 if pixel_val > img_gray_tmp.mean() else 0

    return img_gray_tmp.reshape(image_gray.shape[0], image_gray.shape[1])

def main():
    images_arr = ["lake.jpeg"]
    for img_name in images_arr:
        image = plt.imread(img_assets + img_name)
        image_gray = rgb2gray(image)
        image_gray = black_and_white_segmentation(image_gray)
        print(image.shape)
        plt.imshow(image_gray, cmap="gray")
        plt.waitforbuttonpress()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Exception: ", str(e))
