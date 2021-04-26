
import cv2
import numpy as np

path= '../data/7.jpg'


        
def main():
    img= cv2.imread(path)
    rotate(img)
    cv2.waitKey()

if __name__ == '__main__':
    main()