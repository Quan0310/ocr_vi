import cv2 as cv
import numpy as np
import random
def augment_image(img):
    rows, cols = img.shape[:2]

    # Xoay ±3 độ
    angle = random.uniform(-3, 3)
    M_rot = cv.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # Dịch ngang và dọc ±5 px
    tx = random.uniform(-5, 5)
    ty = random.uniform(-5, 5)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])

    # Shear ±2 độ
    shear = random.uniform(-2, 2) * np.pi / 180
    M_shear = np.float32([[1, np.tan(shear), 0], [0, 1, 0]])

    # Kết hợp các biến đổi
    img = cv.warpAffine(img, M_rot, (cols, rows), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
    img = cv.warpAffine(img, M_trans, (cols, rows), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))
    img = cv.warpAffine(img, M_shear, (cols, rows), flags=cv.INTER_LINEAR, borderValue=(255, 255, 255))

    return img