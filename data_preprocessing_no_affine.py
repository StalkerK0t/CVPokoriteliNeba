import cv2
import numpy as np
import glob
import os


def image_prep(image):
    image = cv2.bilateralFilter(image, 9, 175, 175)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    square_contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[1]
    x, y, w, h = cv2.boundingRect(square_contour)

    mask = np.zeros_like(bin)
    cv2.drawContours(mask, [square_contour], 0, (255, 255, 255), -1)
    cv2.drawContours(mask, [square_contour], 0, (0, 0, 0), 3)
    crop = np.zeros_like(bin)
    crop[mask == 255] = bin[mask == 255]

    delta = 5
    crop = crop[y - delta:y + h + delta, x - delta:x + w + delta]  # crop to size
    crop = cv2.bilateralFilter(crop, 9, 175, 175)
    finish = cv2.resize(crop, (32, 32))

    return finish


paths = []
for i in range(36):
    paths.append(glob.glob(f".\\dataset\\origin\\{i:02d}\\*"))
print(paths)
paths = np.ravel(paths)

for path in paths:
    img = cv2.imread(path)
    print(path)
    if not os.path.exists(".\\dataset\\data_no_affine\\" + path[17:19]):
        os.makedirs(".\\dataset\\data_no_affine\\" + path[17:19])
    try:
        cropped = image_prep(img)
        cv2.imwrite(".\\dataset\\data_no_affine\\" + path[17:], cropped)
    except:
        print("ERROR: " + path)
