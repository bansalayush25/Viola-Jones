import numpy as np

def convert_integral(img):
    row_sum = np.zeros(img.shape)
    integral_img = np.zeros(img.shape)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            row_sum[y, x] = row_sum[y-1, x] + img[y, x] 
            integral_img[y, x] = integral_img[y, x-1] + row_sum[y, x]
    return integral_img

def sum_region(integral_img, top_left, bottom_right):
    top_left = (top_left[1], top_left[0])
    bottom_right = (bottom_right[1], bottom_right[0])
    if top_left == bottom_right:
        return integral_img[top_left]
    top_right = (bottom_right[0], top_left[1])
    bottom_left = (top_left[0], bottom_right[1])
    return integral_img[bottom_right] - integral_img[top_right] - integral_img[bottom_left] + integral_img[top_left]