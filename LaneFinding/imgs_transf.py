import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def Undistort(img):
    # 记载相机内参
    dist_pickle = pickle.load( open("LaneFinding/wide_dist_pickle.p", "rb") )
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    # 恢复图片扭曲
    undist_img = cv2.undistort(img, mtx, dist, None, mtx)
    return undist_img

def PerspectiveTransform(img):
    undist_img = Undistort(img)
    m,n = img.shape[0], img.shape[1] # 
    img_size = (n, m) # 这里注意一下是反的
    # from proj 1 of term 1, oh a funny journey....
    # src = np.float32([[490, 482],[810, 482],
    #                   [1250, 720],[40, 720]])
    # dst = np.float32([[0, 0], [1280, 0], 
    #                  [1250, 720],[40, 720]])
    src = np.float32([[490, 470],[810, 470],
                      [1280, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1280, 720],[40, 720]])
    ##---
    # 透视变换，又称鸟视。。。see car lane using bird eye ^o^
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


