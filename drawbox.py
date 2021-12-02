import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
from cv2 import imread, rectangle


def draw_box(img_path):
    img = imread(img_path)
    h, w = img.shape[:2]
    lab = img * 0
    iter = random.randint(10, 15)
    for i in range(0, iter):
        x2, y2 = random.randint(10, w), random.randint(10, h)
        x1, y1 = random.randint(0, w - x2), random.randint(0, h - y2)
        stroke = random.randint(1, 5)

        rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), stroke)
        rectangle(lab, (x1, y1), (x2, y2), (255, 255, 255), stroke)
    return img, lab


all_imgs = glob.glob('./dataset/DIBCO/img/*.jpeg')
img_out = './dataset/border/img'
lab_out = './dataset/border/gt'

for fname in all_imgs:
    img, lab = draw_box(fname)
    cv2.imwrite(os.path.join(img_out, os.path.basename(fname)), img)
    cv2.imwrite(os.path.join(lab_out, os.path.basename(fname)), ~lab)
