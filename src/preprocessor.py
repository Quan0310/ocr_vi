import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch 
def pad_image(img, target_w, target_h):
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    img_padded = cv.copyMakeBorder(img_resized, top, bottom, left, right,
                                   borderType=cv.BORDER_CONSTANT, value=[255,255,255])
    return img_padded

def preprocess(images, img_w, img_h):
    img = pad_image(images, img_w, img_h)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # [1, H, W] for PyTorch
    return torch.from_numpy(img)


