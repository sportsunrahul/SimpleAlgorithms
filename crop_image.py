import numpy as np

def getRotationMatrix(t):
    return np.array([[np.cos(t),  np.sin(t), 0],\
                     [-np.sin(t), np.cos(t), 0],\
                     [0, 0, 1]])

def getTranlationMatrix(C, w, h):
    T = np.identity(3)

    T[0,2] = C[0] - w//2
    T[1,2] = C[1] - h//2

    return T

def transformCoordinates(x, y, T):
    point = np.array([x, y, 1])
    new_xy = np.matmul(T, point)

    return new_xy[0], new_xy[1]

def f(x, p1, p2):
    return (p2-x)/(p2-p1)

def bilinear_interpolation(image, x, y):
    x1, x2 = int(np.floor(x)), int(np.ceil(x))
    y1, y2 = int(np.floor(y)), int(np.ceil(y))
    p11, p12, p21, p22 = image[x1,y1], image[x1,y2], image[x2,y1], image[x2,y2]
    
    if x1 != x2:
        r1 = f(x,x2,x1)*p11 + f(x,x1,x2)*p12
        r2 = f(x,x2,x1)*p21 + f(x,x1,x2)*p22
    else:
        r1 = (p11+p12)/2
        r2 = (p12+p22)/2
    
    if y1 != y2:
        val = f(y,y2,y1)*r1 + f(y,y1,y2)*r2
    else:
        val = (r1+r2)/2

    return val  

def crop_image(image, C, theta, w, h):
    R = getRotationMatrix(theta)
    T = getTranlationMatrix(C, w, h)
    tranformation_matrix = np.matmul(T,R)

    H, W = image.shape
    cropped_img = np.zeros((w,h))
    list_coord = []
    for y in range(0,h):
        for x in range(0,w):
            x_, y_ = transformCoordinates(x, y, tranformation_matrix)
            list_coord.append((y_,x_))
            if x_>=0 and x_<H-1 and y_>=0 and y_<W-1:
                val = bilinear_interpolation(image, x_, y_)
                cropped_img[y,x] = val

    return cropped_img

if __name__ == "__main__":
    image = np.array([[0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,1,1,0],
                      [0,0,0,0,0,0,0,1,1,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0,0,0]]
                    )
    image = np.arange(60).reshape(6,10)
    print(image)
    cropped_img = crop_image(image, (4,2), 3.14/4, 4, 4)
    print(cropped_img)