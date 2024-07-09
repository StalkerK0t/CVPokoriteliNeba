import cv2
import numpy as np


def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 1920 // 4, 1080 // 4)

image = cv2.imread("dataset/origin/0/image_110.jpg")
show("RGB", image)

image = cv2.bilateralFilter(image,9,175,175)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, bin = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
show("BIN", bin)


contours, hierarchy = cv2.findContours(bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    areas.append((area, cnt))
# print(contours[1])

areas.sort(key=lambda x: x[0], reverse=True)
areas.pop(0) # remove biggest contour
x, y, w, h = cv2.boundingRect(areas[0][1]) # get bounding rectangle around biggest contour to crop to
# img = cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), 2)
crop = bin[y:y+h, x:x+w] # crop to size
crop = cv2.bilateralFilter(crop,9,175,175)
square_contour = areas[0][1]

# coord on big img
a = square_contour[np.argmin(square_contour[:, :, 0]), 0, :]  # top left angle (min by x)
b = square_contour[np.argmin(square_contour[:, :, 1]), 0, :]  # top right angle (min by y)
c = square_contour[np.argmax(square_contour[:, :, 1]), 0, :]  # bottom left angle (max by y)

# coord on small img
a = a[0] - x, a[1] - y
b = b[0] - x, b[1] - y
c = c[0] - x, c[1] - y
print(a, b, c)
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
finish = finish[delta:final_size-delta, delta:final_size-delta]

# print(warp_mat)

cv2.imshow("FINAL", finish)
# print(finish.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()