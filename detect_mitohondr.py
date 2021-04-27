import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt



def viewImage(image): 
    cv2.namedWindow('Green - healthy,Red - damaged', cv2.WINDOW_NORMAL)
    cv2.imshow('Green - healthy,Red - damaged', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def mask_func(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    healthy_low = np.array([0,0,100])                  
    healthy_high = np.array([0, 0, 117]) 
    mask = cv2.inRange(hsv_img, healthy_low, healthy_high)
    return mask

n = 65
filter1 = 3
filter2 = 5
filter3 = 7

filename_initial = '/home/lirkir/current_projects/theUniversity/forLabs/mimohodr/training0000.jpg'
filename_test = '/home/lirkir/current_projects/theUniversity/forLabs/mimohodr/training_groundtruth0000.bmp'
filename_to_save = "/home/lirkir/current_projects/theUniversity/forLabs/mimohodr/result"


image_initial = cv2.imread(filename_initial)
image_test = cv2.imread(filename_test)

result_img1 = image_initial


for i in range(n):
    image_blur = cv2.medianBlur(result_img1,filter1)
    mask = mask_func(image_blur)
    result_img1 = cv2.bitwise_and(image_blur, image_blur, mask=mask)   

plt.subplot(1,3,1)
plt.imshow(result_img1)
    
for i in range(n):
    image_blur = cv2.medianBlur(result_img1,filter2)
    mask = mask_func(image_blur)
    result_img1 = cv2.bitwise_and(image_blur, image_blur, mask=mask)
    
plt.subplot(1,3,2)
plt.imshow(result_img1)

for i in range(n):
    image_blur = cv2.medianBlur(result_img1,filter3)
    mask = mask_func(image_blur)
    result_img1 = cv2.bitwise_and(image_blur, image_blur, mask=mask)
    
plt.subplot(1,3,3)
plt.imshow(result_img1)
plt.show() 
        
    
st = np.copy(image_initial)
st = np.where(result_img1 != 0,st + result_img1,st)


imageio.mimsave(filename_to_save +".gif",[image_initial,st],duration=0.8)
cv2.imwrite(filename_to_save +".bmp", result_img1)

############################## TEST ##########################################

import numpy as np
from sklearn.metrics import jaccard_score
import imageio

im = image_test.ravel()
r, threshold = cv2.threshold(result_img1, 0, 255, cv2.THRESH_BINARY)

im2 = threshold.ravel()


im[im == 255] = 1
im2[im2 == 255] = 1

j = jaccard_score(im, im2)
print(j)



