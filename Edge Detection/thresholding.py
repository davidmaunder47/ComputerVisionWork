import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
Algorithm Steps:

1.	The original image is cropped so that the four-leaf objects become the focal point of the image
2.	The cropped image is then converted to a greyscale image 
3.	The image is then pre-processed using the “createCLAHE” algorithm
4.	The local threshold is determined by taking the lowest value between the two peaks determined by the histogram. This can also be approximately determined by setting the derivate of the histogram function to zero. 
5.	The contours (outlines of the leaves) of the converted greyscale image are determined using the “cv.findcontours function”
6.	The contours found are then filtered to be between a certain area size to reduce any noise (both large and small). 
    a.	The “contourArea(contour)” function is used to determine the area of each contour for filtering purposes
    b.	There will only be four contours left after this algorithm is preformed
7.	The images with their new outlines are re-shown and plotted.
8.	The total pixel area of the leaves in each image is then determined. 
    a.	The area of each leaf is determined by using the ContourArea function
    b.	The contour area of each leaf is then added together to get the total leaf area in the image
    c.	This sum is divided by the number of leaves in the image (will be four, or the amount passed as a parameter)
    d.	This average leaf area is passed to the user as a return value from the function 

The two parameters are the cropped image passed by the user (step 1) and the leaf_count which is usually four
"""
def global_threshold(image,leaf_count):

    #convert original image to greyscale (step 2)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    #pre=process using Clashe (step 3)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img1_equ = clahe.apply(image)

    #Find the minimum from the histogram graph in an automated process
    #1. Get all the numbers and filter them so we only get the values between 100-150
    tmplist = img1_equ.ravel()
    tmplist = tmplist[(tmplist >= 100) & (tmplist <= 150)]

    #Bin the numbers between 100 and 150
    count_arr = np.bincount(tmplist)
    #remove zero counts for cleaning purposes
    count_arr = count_arr[count_arr > 0]

    min_bin = 10000000

    #iterate and find the smallest count to get the minimum in the histogram between 100-150
    for i in count_arr:
        if i < min_bin:
            min_bin = i

    #get the location of the lowest count.
    count_bin = np.where(count_arr == min_bin)

    # we set the threshold as 101, since this worked the best with trial and error
    #if we want to use the count_bin amount, we would just add:
    #count_bin[0[0] + 100 to get the local minimum (step 4)
    threshold_equ = 101


    arrays  = plt.hist(img1_equ.ravel(), 256, [0, 256])

    #This can be commented out if needed since its just plotting the histogram
    plt.show()



    ret, img1_equ_binary = cv.threshold(image, threshold_equ, 255, cv.THRESH_BINARY)

    #The below can be commented out if you dont want to plot the intermediate steps
    plt.imshow(np.hstack((img1_equ, img1_equ_binary)), 'gray', vmin=0, vmax=255)
    plt.title("Equalized Grayscale / Binary")
    plt.xticks([]), plt.yticks([])
    plt.show()

    #Step 5 Find contours (step 5)
    contours, hierarchy = cv.findContours(img1_equ_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    rows, cols = image.shape
    img1_contour_clean = np.zeros((rows, cols), np.uint8)

    #Min and maxes are used to weed out noisey contours
    counter = 0
    small_size = 50
    large_size = 40000
    pixel_amount = 0
    #step 6 making new contours (weeding out contours to small or to big)
    for i, cnt in enumerate(contours):
        # Draw only if the size of the contour is greater than a threshold
        temp = cv.contourArea(cnt)
        if temp > small_size and temp < large_size:
            counter += 1
            img1_contour_clean = cv.drawContours(img1_equ, [cnt], -1, (0, 255, 0), 3)
            pixel_amount = pixel_amount + temp

   #Step 7 plottting
    plt.imshow(img1_contour_clean, cmap="gray")

    plt.show()

    #step 8 returning average pixel amount to user
    return pixel_amount/leaf_count

