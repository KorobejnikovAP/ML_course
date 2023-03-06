"""
 Copyright (c) 2023 ML course
 Created by Aleksey Korobeynikov
"""

import cv2
import numpy as np
import math


INTERPOLATION_TYPES = {
    'LINEAR': cv2.INTER_LINEAR,
    'CUBIC': cv2.INTER_CUBIC,
    'NEAREST': cv2.INTER_NEAREST,
    'AREA': cv2.INTER_AREA,
}


def resize_image(image, size, keep_aspect_ratio=False, interpolation=cv2.INTER_LINEAR):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size, interpolation=interpolation)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized_frame


def resize_image_with_aspect(image, size, interpolation=cv2.INTER_LINEAR):
    return resize_image(image, size, keep_aspect_ratio=True, interpolation=interpolation)


def pad_image(image, size):
    h, w = image.shape[:2]
    if h != size[1] or w != size[0]:
        image = np.pad(image, ((0, size[1] - h), (0, size[0] - w), (0, 0)),
                               mode='constant', constant_values=0)
    return image


def resize_image_letterbox(image, size, interpolation=cv2.INTER_LINEAR):
    ih, iw = image.shape[0:2]
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh), interpolation=interpolation)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    resized_image = np.pad(image, ((dy, dy + (h - nh) % 2), (dx, dx + (w - nw) % 2), (0, 0)),
                           mode='constant', constant_values=0)
    return resized_image


def crop_resize(image, size):
    desired_aspect_ratio = size[1] / size[0] # width / height
    if desired_aspect_ratio == 1:
        if (image.shape[0] > image.shape[1]):
            offset = (image.shape[0] - image.shape[1]) // 2
            cropped_frame = image[offset:image.shape[1] + offset]
        else:
            offset = (image.shape[1] - image.shape[0]) // 2
            cropped_frame = image[:, offset:image.shape[0] + offset]
    elif desired_aspect_ratio < 1:
        new_width = math.floor(image.shape[0] * desired_aspect_ratio)
        offset = (image.shape[1] - new_width) // 2
        cropped_frame = image[:, offset:new_width + offset]
    elif desired_aspect_ratio > 1:
        new_height = math.floor(image.shape[1] / desired_aspect_ratio)
        offset = (image.shape[0] - new_height) // 2
        cropped_frame = image[offset:new_height + offset]

    return cv2.resize(cropped_frame, size)

RESIZE_TYPES = {
    'crop' : crop_resize,
    'standard': resize_image,
    'fit_to_window': resize_image_with_aspect,
    'fit_to_window_letterbox': resize_image_letterbox,
}
