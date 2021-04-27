import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio

def viewImage(image): 
    cv2.namedWindow('Green - healthy,Red - damaged', cv2.WINDOW_NORMAL)
    cv2.imshow('Green - healthy,Red - damaged', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def damaged_area_mask(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dam_low0 = np.array([10,50,50])                  
    dam_high0 = np.array([40, 255, 255])             
    dam_low1 = np.array([90,50,50])  
    dam_high1 = np.array([100, 255, 255]) 
    damaged_mask = cv2.inRange(hsv_img, dam_low0, dam_high0) + cv2.inRange(hsv_img, dam_low1, dam_high1) # формирование маски
    return damaged_mask

def healthy_area_mask(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    healthy_low = np.array([40,0,0])                  
    healthy_high = np.array([90, 255, 255]) 
    healthy_mask = cv2.inRange(hsv_img, healthy_low, healthy_high) # формирование маски
    return healthy_mask

def resulting_mask(image): 
    resulting_mask = healthy_area_mask(image) + damaged_area_mask(image)
    return resulting_mask
    

def display(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)     # перевод из цветового пространства BGR в HSV
    damag_mask = damaged_area_mask(image)
    healt_mask = healthy_area_mask(image)
    reslt_mask = resulting_mask(image)
    hsv_img[reslt_mask == 0] = ([0,0,0])
    hsv_img[healt_mask > 0] = ([0,100,100]) 
    hsv_img[damag_mask > 0] = ([150,255,200])
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    plt.subplot(1,2,1)
    plt.imshow(rgb_img)
    plt.subplot(1,2,2)
    plt.imshow(image)
    plt.show()
    viewImage(image)
    viewImage(rgb_img)
    

def save(image,directory_to_save,name_file):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cp_img = np.copy(image)

    damag_mask = damaged_area_mask(image)  
    healt_mask = healthy_area_mask(image)
    reslt_mask = resulting_mask(image)
    
    hsv_img[reslt_mask == 0] = ([0,0,0])     
    hsv_img[healt_mask > 0] = ([180,255,255]) 
    hsv_img[damag_mask > 0] = ([255,100,100]) 
    
    
    
    #cv2.imwrite(directory_to_save+name_file, rgb_img)
    #rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    
    new_file_name = name_file.split(".")[0] + ".gif"
    file_name = directory_to_save + new_file_name
    hsv_img = np.where(hsv_img != ([0,0,0])  ,cp_img*0.5 + hsv_img*0.5,cp_img)

    
    #rgb_img = cv2.cvtColor(cp_img, cv2.COLOR_HSV2RGB)
    
    imageio.mimsave(file_name,[cp_img,hsv_img],duration=0.8)
          

    
def square_contour(image,mask):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_img[mask == 0] = ([0,0,0])
    hsv_img[mask > 0] = ([75,255,200])
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(gray_img, 90, 255, 0)
    contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    square_area = 0
    for contour in contours:
        square_area += cv2.contourArea(contour)
    return square_area

def square_pixel(image,mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_bit = cv2.bitwise_and(image, image, mask=mask)
    return len(image_bit[image_bit>0])/len(image_bit)*100


def square_watershed_method(image,directory_to_save,file_name):
    markers = np.zeros((image.shape[0],image.shape[1]),dtype="int32")
    markers[125:128,100:200] = 2
    markers[0:256,0:10] = 1
    markers[0:20,-50:0] = 1
    markers[230:256,-50:0] = 1

    mask  = cv2.watershed(image,markers)    
    square = list(mask.ravel()).count(2)
    
    new_file_name = file_name.split(".")[0] + ".gif"
    file_name = directory_to_save + new_file_name
    mask[mask == -1] = 0
    mask[mask == 2] = 255
    mask = mask.astype(np.uint8)

    st = np.copy(image)
    st[:,:,0] = np.where(mask == 255,st[:,:,0]*0.5 + 127,st[:,:,0])
    st[:,:,1] = np.where(mask == 255,st[:,:,1]*0.5 + 0,st[:,:,1])
    st[:,:,2] = np.where(mask == 255,st[:,:,2]*0.5 + 0,st[:,:,2])
    imageio.mimsave(file_name,[image,st],duration=0.8)
    
    return square
    
    

    
#########################TESTS##########################
# блок тестирования на 10 картинках 
directory = '/home/lirkir/current_projects/theUniversity/forLabs/test_cheets/' 
directory_to_save_jpg = '/home/lirkir/current_projects/theUniversity/forLabs/jpg/'
directory_to_save_gif = '/home/lirkir/current_projects/theUniversity/forLabs/gif/'

files = os.listdir(directory) 

for file in files:
    image0 = cv2.imread(directory+file)
    #image0 = cv2.medianBlur(image0,9)
    path_name_mask = 'mask'+file
    save(image0,directory_to_save_jpg,path_name_mask)
    sheet_mask = resulting_mask(image0)
    damaged_mask = damaged_area_mask(image0)

    #display(image)
    
    image1 = imageio.imread(directory+file)
    path_name_gif = 'gif'+file
    square_watershed = square_watershed_method(image1,directory_to_save_gif,path_name_gif)

    
    
    print(file,')')
    print("Contour method.")   
    print("square_contour:",square_contour(image0,sheet_mask))
    print("sheet damage percentage:",square_contour(image0,damaged_mask)/square_contour(image0,sheet_mask)*100)
    print()
    print("Pixel method.")   
    print("square_pixel:",square_pixel(image0,sheet_mask))
    print("sheet damage percentage:",square_pixel(image0,damaged_mask)/square_pixel(image0,sheet_mask)*100)
    print()
    print("Watershed method.")
    print("square_Watershed:",square_watershed)
    print()
    print()
