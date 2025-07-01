# image_utils.py

import cv2

def resize_with_aspect_ratio(img, max_width=800):
    h, w = img.shape
    scale = max_width / w if w > h else max_width / h
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def rotate_image(img, angle):
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def pad_to_same_height(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 == h2:
        return img1, img2
    max_h = max(h1, h2)

    def pad(img, target_h):
        diff = target_h - img.shape[0]
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)

    return pad(img1, max_h), pad(img2, max_h)
