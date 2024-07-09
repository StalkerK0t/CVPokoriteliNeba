import cv2
import numpy as np
import glob
import os


def image_prep(image):
    image = cv2.bilateralFilter(image, 9, 175, 175)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours.pop(0)  # remove biggest contour
    square_contour = contours[0]
    x, y, w, h = cv2.boundingRect(square_contour)
    crop = bin[y:y + h, x:x + w]
    crop = cv2.bilateralFilter(crop, 9, 175, 175)

    # coord on big img
    a = square_contour[np.argmin(square_contour[:, :, 0]), 0, :]  # top left angle (min by x)
    b = square_contour[np.argmin(square_contour[:, :, 1]), 0, :]  # top right angle (min by y)
    c = square_contour[np.argmax(square_contour[:, :, 1]), 0, :]  # bottom left angle (max by y)

    # coord on small img
    a = a[0] - x, a[1] - y
    b = b[0] - x, b[1] - y
    c = c[0] - x, c[1] - y

    img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    for point in (a, b, c):
        # print(point)
        cv2.circle(img, point, 5, (0, 0, 255), 10)

    delta = 10
    final_size = 128 + 2 * delta
    src = np.array([a, b, c]).astype(np.float32)
    dst = np.array([[0, 0], [final_size, 0], [0, final_size]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(src, dst)

    finish = cv2.warpAffine(crop, warp_mat, (crop.shape[1], crop.shape[0]))
    finish = finish[delta:final_size - delta, delta:final_size - delta]

    return finish

paths = []
for i in range(36):
    paths.append(glob.glob(f".\\dataset\\origin\\{i}\\*"))
paths = np.ravel(paths)

for path in paths:
    img = cv2.imread(path)
    print(path)
    cropped = image_prep(img)

    if not os.path.exists(".\\dataset\\data\\" + path[17:19]):
        os.makedirs(".\\dataset\\data\\" + path[17:19])
    cv2.imwrite(".\\dataset\\data\\" + path[17:], cropped)
