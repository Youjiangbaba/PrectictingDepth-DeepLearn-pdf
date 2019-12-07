import cv2
import numpy
ww = 600
hh = 800
w = ww
h = hh

def size_int(img,flag):
    global w,h,ww,hh
    if flag == 1:
        w = img.shape[1]
        h = img.shape[0]
        while 1:
            if w >1500 or h > 1200:
                w = w //2
                h = h//2
            else:
                break
        
    elif flag == 0:
        w = ww
        h = hh