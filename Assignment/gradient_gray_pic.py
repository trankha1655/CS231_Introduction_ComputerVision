
import numpy as np
import cv2
#import matplotlib.pylot as plt

def gd(img):
    clone= np.ones_like(img)
    shape= img.shape
    for i in range(shape[0]-1):
        for j in range(shape[1]-1):
            x= img[i+1][j]- img[i][j] 
            y= img[i][j+1]- img[i][j]
            clone[i][j]= (x**2+y**2)
    return clone

if __name__=="__main__":
    img_path= '7.jpg'
    img= cv2.imread(img_path,0)
    cv2.imshow('normal',img)
    cv2.waitKey()
    gd_img= gd(img)
    cv2.imshow('gd',gd_img)
    cv2.waitKey()
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    lap = np.uint8(np.absolute(lap))
    cv2.imshow('lap',lap)
    cv2.waitKey()