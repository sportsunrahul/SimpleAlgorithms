import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from glob import glob

def conv2D(img, kernel):
    h, w = img.shape
    kernel = kernel[::-1,::-1]
    k = len(kernel)
    img_copy = np.zeros((h+k-1, w+k-1))
    img_copy[k//2:-(k//2), k//2:-(k//2)] = np.copy(img)
    res = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            res[i,j] = (img_copy[i:i+k,j:j+k] * kernel).sum()
    return res
    
def gaussian_blur(img, k, sigma=1.4):
    xi, yi = np.meshgrid(np.linspace(-(k//2),k//2,k),np.linspace(-(k//2),k//2,k))
    kernel = np.exp(-(xi**2 + yi**2)/(2*(sigma**2))) / ((2*np.pi)*sigma*sigma)
    return conv2D(img, kernel)

def sobel_filter(img):
    kx = np.array([[-1.,0.,1.],
                  [-2.,0.,2.],
                  [-1.,0.,1.]],np.float32)
    ky = np.array([[1.,2.,1.],
                  [0.,0.,0.],
                  [-1.,-2.,-1.]],np.float32)
    gx = conv2D(img,kx)
    gy = conv2D(img,ky)
    
    
    g = np.hypot(gx, gy)
    theta = np.arctan2(gy,gx)
    g = g/g.max() *255
    return gx, gy, g, theta

def nms(img, theta):
    angle = 180 * theta / np.pi
    angle[angle < 0] += 180
    h, w = img.shape
    Z = np.zeros((h,w))
    
    for i in range(1,h-1):
        for j in range(1,w-1):
            q, r = 255, 255
            if (0 <= angle[i,j] < 22.5) or 157.5 <= angle[i,j] < 180:
                q, r = img[i,j+1], img[i,j-1]
            elif 22.5 <= angle[i,j] < 67.5:
                q, r = img[i+1,j-1], img[i-1,j+1]
            elif 67.5 <= angle[i,j] < 112.5:
                q, r = img[i+1,j], img[i-1,j]
            elif 112.5 <= angle[i,j] < 157.5:
                q, r = img[i-1,j-1], img[i+1,j+1]
            
            if img[i,j] >= q and img[i,j] >= r:
                Z[i,j] = img[i,j]
    return Z


def threshold(img, lowThresholdRatio = 0.5, highThresholdRatio = 0.9):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    print(highThreshold, lowThreshold)
    h, w = img.shape
    res = np.zeros((h,w), dtype = np.int32)
    weak, strong = 25, 255
    
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img >= lowThreshold) & (img < highThreshold))
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    h, w = img.shape
    for i in range(1, h-1):
        for j in range(1, w-1):
            if img[i,j] == weak:
                if img[i,j-1] == strong or img[i][j+1] == strong or img[i-1][j-1] == strong or img[i-1][j] == strong or img[i-1][j+1] == strong or img[i+1][j-1] == strong or img[i+1][j] == strong or img[i+1][j+1] == strong:
                    img[i,j] = strong
                else:
                    img[i,j] = 0

    return img


    def main(images):

    def show(img):
        show.count += 1
        plt.figure(show.count)
        plt.imshow(img.astype(int), cmap='gray')
    show.count = 0

    for idx, image in enumerate(images):
        org_img = cv2.imread(image,0)

        img = gaussian_blur(org_img,5,1.4)
        show(np.hstack((org_img, img)))

        gx, gy, g, theta = sobel_filter(img)
        show(np.hstack((img,g)))


        img = nms(g, theta)
        show(np.hstack((g,img)))

        th, weak, strong = threshold(img,lowThresholdRatio=0.5, highThresholdRatio=0.2)
        show(np.hstack((img,th)))

        img = hysteresis(np.copy(th), weak)
        show(np.hstack((th,img)))

        show(np.hstack((org_img, img)))
        plt.savefig('edge{}.png'.format(idx),dpi=400)
        plt.show()

if __name__ == "__main__":
    images = glob('*.jpg')
    main(images)