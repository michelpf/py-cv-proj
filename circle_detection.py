# python version: 3.6
# opencv version: 3.30

# Notes:
#
#    The main approach is to clean images before start the object detection. In the first image there is no noise, so
# was not applied any kind of noise removal or reduction. In the second image, there are patterns which intefere in the
# circle detecction. In that case, the noise removal was needed.
#   Using simple techniques of erosion and dilation was possible to clean the second image and procede to circle detecction.
#   Since the challenge is to detect circles not segment them was not invest time to proper segment each circle. The goal
# is to detect the right circles in each images.
#
# Use: something like one of (-c for clean imagens, -n for noisy images)
#    python circle_detecction.py -c images/image_filename
#    python3 circle_detecction.py -n images/image_filename


import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys, getopt

def image_detecction_clean(image_path='imagens/circles.png'):

    print("[CLEAN] Starting circle detection...")

    image = cv2.imread(image_path)
    image_original = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.medianBlur(gray, 3)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 45, param1=190, param2=14.5, minRadius=5)

    i_circle = 0
    for i in circles[0,:]:
        i_circle+=1
        radius = i[2];
        print("Circle detected " + str(i_circle) + ", diameter " + str(2*radius))
        cv2.circle(image,(i[0], i[1]), i[2], (255, 0, 0), 2)

    f = plt.figure(figsize=(16, 4))
    f.add_subplot(1,2, 1)
    plt.imshow(image_original)
    f.add_subplot(1,2, 2)
    plt.imshow(image)
    plt.suptitle('Comparsion (Circles detected ' + str(i_circle)+ ')')
    plt.show(block=True)

    print("End of detection.")

def image_detecction_noisy(image_path='imagens/shapes_leo.jpg'):

    print("[NOISY] Starting circle detection...")
    image_original = cv2.imread(image_path)
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((4, 4), np.uint8)

    erosao = cv2.erode(gray, kernel, iterations=2)
    dilatacao = cv2.dilate(erosao, kernel, iterations=2)

    blur = cv2.medianBlur(erosao, 3)

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 45, param1=190, param2=40, minRadius=5)

    i_circle = 0
    for i in circles[0, :]:
        i_circle += 1
        radius = i[2];
        print("Circle detected " + str(i_circle) + ", diameter " + str(2 * radius))
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 2)

    f = plt.figure(figsize=(16, 4))
    f.add_subplot(1, 2, 1)
    plt.imshow(image_original)
    f.add_subplot(1, 2, 2)
    plt.imshow(image)
    plt.suptitle('Comparsion (Circles detected ' + str(i_circle) + ')')
    plt.show(block=True)

    print("End of detection.")


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"cn:")
    except getopt.GetoptError:
        print('circle_detection.py [-n <for noisy images> | -c <for clean images> ] <image path>')
        sys.exit(2)

    if len(opts) == 0:
        print('circle_detection.py [-n <for noisy images> | -c <for clean images> ] <image path>')

    for opt, arg in opts:
        if opt == '-c':
            image = args[0]
            image_detecction_clean(image)

        if opt == '-n':
            image = arg
            image_detecction_noisy(image)


if __name__=="__main__":
    main(sys.argv[1:])