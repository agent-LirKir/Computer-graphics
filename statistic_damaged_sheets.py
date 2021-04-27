import cv2
import os 
import numpy as np
import functions_defect_detection as d
   
directory = '/home/lirkir/current_projects/theUniversity/forLabs/sheets/' 
files = os.listdir(directory) 

S_damaged_area_contour = []
S_whole_area_contour = []
S_damaged_area_pixel = []
S_whole_area_pixel = []
S_whole_area_whatershed = []

for file in files:
    
    image = cv2.imread(directory+file)
    
    print(directory+file)
    
    damaged_mask = d.damaged_area_mask(image)
    sheet_mask = d.resulting_mask(image)
    
    S_damaged_area_contour.append(d.square_contour(image,damaged_mask))
    S_whole_area_contour.append(d.square_contour(image,sheet_mask))
    
    S_damaged_area_pixel.append(d.square_pixel(image,damaged_mask))
    S_whole_area_pixel.append(d.square_pixel(image,sheet_mask))
    
    S_whole_area_whatershed.append(d.square_watershed_method(image))


# определение среднего значения по выборкам из директории

averageSquareContour = sum(S_whole_area_contour)/len(files)
averagePercentageContour = sum(np.array(S_damaged_area_contour)/np.array(S_whole_area_contour)*100)/len(files)

averageSquarePixel = sum(S_whole_area_pixel)/len(files)
averagePercentagePixel = sum(np.array(S_damaged_area_pixel)/np.array(S_whole_area_pixel)*100)/len(files)

averageSquareWhatershed = sum(S_whole_area_whatershed)/len(files)

print()
print("Square pixel method")
print("average square of whole sheet:",averageSquarePixel)
print("average sheet damage percentage:",averagePercentagePixel)
print()
print("Square contour method")
print("average square of whole sheet:",averageSquareContour)
print("average sheet damage percentage:",averagePercentageContour)
print()
print("Square whatershed method")
print("average square of whole sheet",averageSquareWhatershed)