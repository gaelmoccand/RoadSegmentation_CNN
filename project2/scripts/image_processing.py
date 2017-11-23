# -*- coding: utf-8 -*-
from skimage.feature import corner_harris, corner_peaks


def image_corner(image, min_distance=5):
    harris = corner_peaks(corner_harris(image), min_distance)
    return harris