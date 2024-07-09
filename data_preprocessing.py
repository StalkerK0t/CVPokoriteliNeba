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

    rect = cv2.minAreaRect(square_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # coord on big img
    a = box[1] + [-delta, -delta]  # top left angle
    b = box[2] + [+delta, -delta]  # top right angle
    c = box[0] + [-delta, +delta]  # bottom left angle

    # coord on small img
    a = a[0] - x, a[1] - y
    b = b[0] - x, b[1] - y
    c = c[0] - x, c[1] - y

    delta = 1
    final_size = 32 + 2 * delta
    src = np.array([a, b, c]).astype(np.float32)
    dst = np.array([[0, 0], [final_size, 0], [0, final_size]]).astype(np.float32)
    warp_mat = cv2.getAffineTransform(src, dst)

    finish = cv2.warpAffine(crop, warp_mat, (crop.shape[1], crop.shape[0]))
    finish = finish[delta:final_size - delta, delta:final_size - delta]

    kernel = np.ones((3, 3), np.uint8)
    finish = cv2.morphologyEx(finish, cv2.MORPH_CLOSE, kernel)
    return finish


paths = []
for i in range(36):
    paths.append(glob.glob(f".\\dataset\\origin\\{i}\\*"))
paths = np.ravel(paths)

for path in paths:
    img = cv2.imread(path)
    print(path)
    if not os.path.exists(".\\dataset\\data\\" + path[17:19]):
        os.makedirs(".\\dataset\\data\\" + path[17:19])
    try:
        cropped = image_prep(img)
        cv2.imwrite(".\\dataset\\data\\" + path[17:], cropped)
    except:
        print("ERROR: " + path)



