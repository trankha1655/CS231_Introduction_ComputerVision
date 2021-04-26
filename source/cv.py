import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import PIL
import os
import numpy as np

class cv:
    def __init__(seft):
        pass

    def imread(path,gray='False'):

        try:
            img = mpimg.imread(path)
            if not gray:
                return img
            else:
                r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
                img_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                return img_gray
        except:
            print('No file such as: {}'.format(path))


    def imshow(img, gray='False',tilte=''):
        if not gray:
            plt.imshow(img)
            plt.tilte(tilte)
        else:
            plt.imshow(img,'gray')
            plt.tilte(tilte)


    def imsave(img,file_name):
        plt.imsave(file_name,img)        

    def flip_horizon(img):
        return img[:][::-1]

    def flip_vertical(img):
        return im[::-1][:]
        
    """def rotate(img):
        h= img.shape[0]
        w= img.shape[1]
        emptyF = np.zeros((w,h),dtype="uint8")
        emptyB = np.zeros((w,h),dtype="uint8")
        emptyBB = np.zeros((w,h),dtype="uint8")


        for i in range(w,radians):
            for j in range(h):
                temp = img[i,j]
                #forward mapping
                xf = (i-330)*math.cos(self.radians)-(j-self.y)*math.sin(self.radians)+self.x
                yf = (i-self.x)*math.sin(self.radians)+(j-self.y)*math.cos(self.radians)+self.x
                #backward mapping should change the forward mapping to the original image
                xbb = (i-self.x)*math.cos(self.radians)+(j-self.y)*math.sin(self.radians)+self.x
                ybb = -(i-self.x)*math.sin(self.radians)+(j-self.y)*math.cos(self.radians)+self.x
                xbb = int(xbb)
                ybb = int(ybb)
                if xf < 660 and yf < 660 and xf>0 and yf > 0:
                    emptyF[int(xf),int(yf)] = temp
                else:
                    pass
                if xbb < 660 and ybb < 660 and xbb>0 and ybb > 0:
                    emptyBB[(xbb),(ybb)] = temp
                else:
                    pass
        cv2.imshow('Forward', emptyF)
        cv2.imshow('Backward', emptyBB)

    """