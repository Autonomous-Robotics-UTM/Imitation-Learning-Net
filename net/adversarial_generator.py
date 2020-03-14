import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('stop.jpg')
data = cv2.imread('000046.jpg')
rows,cols,ch = img.shape

dimensions = (1200,1200)
max_x,max_y = dimensions

scale_x = 2
scale_y = 2

offset_x = 500
offset_y = -100

mid_x = 500 + offset_x
mid_y = 500 + offset_y




x_bu = mid_x - 100/2 * scale_x
x_bd = mid_x - 100/2 * scale_x

y_bu = mid_y - 100/2 * scale_y
y_bd = mid_y + 100/2 * scale_y


x_eu = mid_x + 100/2 * scale_x
x_ed = mid_x + 100/2 * scale_x

y_eu = mid_y - 150/2 * scale_y
y_ed = mid_y + 150/2 * scale_y

#[top_left,top_right,bottom_left,bottom_right]
pts1 = np.float32([[0,0],[max_x,0],[0,max_y],[max_x,max_y]])
#pts2 = np.float32([[50,50],[250,0],[50,250],[250,300]])


pts2 = np.float32([[400.0, 450.0], [600.0, 400.0], [400.0, 550.0], [600.0, 600.0]])
#pts2 = np.float32([[300,50],[700,0],[300,250],[700,300]])

pts2 = np.float32([[x_bu,y_bu],[x_eu,y_eu],[x_bd,y_bd],[x_ed,y_ed]])
#print([[x_bu,y_bu],[x_eu,y_eu],[x_bd,y_bd],[x_ed,y_ed]])
#pts2 = np.float32([[x_bu,y_bu],[x_eu,y_eu],[x_bd,y_bd],[x_ed,y_ed]])


M = cv2.getPerspectiveTransform(pts1,pts2)

#dst = cv2.warpPerspective(img,M,(1200,1200))

img1 = img
noise = np.random.rand(1200,1200,3)
mask = img1==[0,0,0]
noise[mask] = img1[mask]


mask = np.random.rand(1200,1200) > 0.5
comb_img = img1.copy()
comb_img[mask] = noise[mask]*255
comb_img = comb_img.reshape(1200,1200,3)

M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(comb_img,M,(1200,1200))
dim = (640,480)
dst_r = cv2.resize(dst, dim)
mask = dst_r > 0
data[mask] = dst_r[mask]



#cv2.imshow('IMAGE',img1)
plt.subplot(141),plt.imshow(img1),plt.title('Input')
plt.subplot(142),plt.imshow(comb_img),plt.title('Noise')
plt.subplot(143),plt.imshow(dst_r),plt.title('Scale+Position')
plt.subplot(144),plt.imshow(data),plt.title('Output')
plt.show()


