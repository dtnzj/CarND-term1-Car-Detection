import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
# from imgs_transf import PerspectiveTransform


def color_threshold(img, s_thresh=(170, 255), \
                    l_thresh=(170, 255), b_thresh=(170, 255)):
    """
    三种不同颜色空间的阈值
    HLS: 色相hue, 饱和度saturation,亮度lightness/luminance
    LUV: 
    Lab: L像素的亮度 a红色到绿色的范围 b黄色到蓝色的范围
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    s_channel = hls[:,:,2]
    l_channel = luv[:,:,0]
    b_channel = lab[:,:,2]

    s_bin = np.zeros_like(s_channel)
    s_bin[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    l_bin = np.zeros_like(l_channel)
    l_bin[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    b_bin = np.zeros_like(b_channel)
    b_bin[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
    return s_bin, l_bin, b_bin

def Processing(img):
    """
    调整函数参数，对Sobel梯度与颜色阈值进行组合
    """
    s_bin, l_bin, b_bin = color_threshold(img, s_thresh=(180, 255), \
                          l_thresh=(225, 255), b_thresh=(155, 200))
    combined_bin = np.zeros_like(s_bin)
    combined_bin[(l_bin == 1) | (b_bin == 1)] = 1
    return combined_bin
