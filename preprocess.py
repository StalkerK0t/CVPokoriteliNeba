import cv2
import numpy as np


def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 1920 // 4, 1080 // 4)

image = cv2.imread("dataset/origin/0/image_001.jpg")
show("RGB", image)

image = cv2.bilateralFilter(image, 9, 175, 175)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, bin = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
# show("Bin", bin)


contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
square_contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[1]
x, y, w, h = cv2.boundingRect(square_contour)

mask = np.zeros_like(bin)
cv2.drawContours(mask, [square_contour], 0, (255, 255, 255), -1)
cv2.drawContours(mask, [square_contour], 0, (0, 0, 0), 5)
crop = np.zeros_like(bin)
crop[mask == 255] = bin[mask == 255]


delta = 10
crop = crop[y-delta:y+h+delta, x-delta:x+w+delta] # crop to size
crop = cv2.bilateralFilter(crop, 9, 175, 175)
# show("Crop", crop)

rect = cv2.minAreaRect(square_contour)
box = cv2.boxPoints(rect)
box = np.intp(box)

a = box[1] + [-delta, -delta]  # top left angle
b = box[2] + [+delta, -delta]  # top right angle
c = box[0] + [-delta, +delta]  # bottom left angle
# print(box)
# cv2.circle(image, box[0], 5, (0, 0, 255), 10)  # bottom left / red
# cv2.circle(image, box[1], 5, (0, 255, 0), 10)  # top left / green
# cv2.circle(image, box[2], 5, (255, 0, 0), 10)  # top right / blue
# cv2.drawContours(image, [box], 0, (0, 0, 255), 5)
# show("Rect", image)


# coord on small img
a = max(a[0] - x + delta, 0), max(a[1] - y + delta, 0)
b = max(b[0] - x + delta, 0), max(b[1] - y + delta, 0)
c = max(c[0] - x + delta, 0), max(c[1] - y + delta, 0)
# print(a, b, c)
# img = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
# for i, point in enumerate((a, b, c)):
#     # print(point)
#     cv2.circle(img, point, 5, (0, 0, 75*i), 10)
# show("Img", img)



final_size = 32
src = np.array([a, b, c]).astype(np.float32)
dst = np.array([[0, 0], [final_size, 0], [0, final_size]]).astype(np.float32)
warp_mat = cv2.getAffineTransform(src, dst)

finish = cv2.warpAffine(crop, warp_mat, (crop.shape[1], crop.shape[0]))
finish = finish[:final_size, :final_size]

kernel = np.ones((3,3),np.uint8)
finish = cv2.morphologyEx(finish, cv2.MORPH_CLOSE, kernel)
# print(warp_mat)

cv2.imshow("Final", finish)
# print(finish.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()