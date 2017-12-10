import numpy as np
import cv2

class Box:
    def __init__(self):
        self.c = float() # confidence
        self.prob = float() # 概率
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.class_num = 0

# ------------------------------------------
# 
# ref https://github.com/sunshineatnoon/Darknet.keras/blob/master/RunTinyYOLO.py
def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2
    l2 = x2 - w2 / 2
    left = max(l1, l2)
    r1 = x1 + w1 / 2
    r2 = x2 + w2 / 2
    right = min(r1, r2)
    return right - left

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0: return 0
    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)
# -------------------------------------------------------------------------------------

def network_to_boxes(network_out, threshold = 0.2, sqrt=1.8, classes=20, num=2, side=7):
    """
    输入 神经网络softmax系数
    输出 
    """
    class_num = 6
    boxes = []
    prob_size = side * side * classes # 输出向量的probability部分的长度=980
    conf_size = side * side * num # 置信度confidence部分的长度=98
    # 确定output vector中的三部分信息  
    probability = network_out[0 : prob_size].reshape([side*side, classes])
    confidence = network_out[prob_size : (prob_size + conf_size)].reshape([side*side, num])
    coordinates = network_out[(prob_size + conf_size) : ].reshape([side*side, num, 4])
    
    # 
    for grid in range(side*side):
        for n in range(num):
            bx   = Box()
            bx.c =  confidence[grid, n]
            bx.x = (coordinates[grid, n, 0] + grid %  side) / side
            bx.y = (coordinates[grid, n, 1] + grid // side) / side
            bx.w =  coordinates[grid, n, 2] ** sqrt 
            bx.h =  coordinates[grid, n, 3] ** sqrt
            p_idx = probability[grid, :] * bx.c
            
            if p_idx[class_num] >= threshold:
                bx.prob = p_idx[class_num]
                boxes.append(bx)
                
    # 处理重叠窗口
    for i in range(len(boxes)):
        box_i = boxes[i]
        if box_i.prob == 0: continue
        for j in range(i + 1, len(boxes)):
            box_j = boxes[j]
            if box_iou(box_i, box_j) >= .4:
                boxes[j].prob = 0.
    return [box for box in boxes if box.prob > 0]

def boxes_drawing(img, boxes, cropped_area=[[500,1280],[300,650]]):
    copy_img = np.copy(img)
    [xmin,xmax] = cropped_area[0]
    [ymin,ymax] = cropped_area[1]
    # 
    #
    for b in boxes:
        h, w, _ = copy_img.shape
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        left = int(left*(xmax-xmin)/w + xmin)
        right = int(right*(xmax-xmin)/w + xmin)
        top = int(top*(ymax-ymin)/h + ymin)
        bot = int(bot*(ymax-ymin)/h + ymin)

        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1
        cv2.rectangle(copy_img, (left, top), (right, bot), (255,255,0), 6)

    return copy_img
