import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


#Canny algorithm
"""
1.	The original image is cropped so that the four-leaf objects become the focal point of the image
2.	The cropped image is then converted to greyscale
3.	The edges (perimeter of the leaf objects) of the greyscale object is determined by the “cv.canny algorithm”
4.	The contours (outlines of the leaves) of the converted greyscale image are determined using the “cv.findcontours function”
5.	The contours found are then filtered to be between a certain size to reduce any noise (both large and small). 
    a.	The filtering is conducted using the contour Area function. 
6.	The images with their new outlines are re-shown and plotted.
7.	The total pixel area of the leaves in each image is then determined. 
    a.	The area of each leaf is determined by using the ContourArea function
    b.	The contour area of each leaf is then added together to get the total leaf area in the image
    c.	This sum is divided by the number of leaves in the image (will be four, or the amount passed as a parameter)
    d.	This average leaf area is passed to the user as a return value from the function 

We only have one parameter (the cropped imaged) we just set the leaf count to be four for this algorithm
"""
def canny (image1):
    #Step 2 convert to bianry
    img = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

    #Step 3 get edges
    edges = cv.Canny(img,100,105)

    #find contours Step 4
    cnts, hier = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rows, cols = img.shape
    img1_contour = cv.drawContours(img, cnts, -1, (0, 255, 0), 3)

    img1_contour_clean = np.zeros((rows, cols), np.uint8)
    small_size = 40

    #Step 5 Filter contours
    counter = 0
    area = 0
    for i, cnt in enumerate(cnts):
        # Draw only if the size of the contour is greater than a threshold
        if cv.contourArea(cnt) > small_size:
            img1_contour_clean = cv.drawContours(edges, [cnt], 0, (0, 255, 0), 3)
            counter+=1
            area = cv.contourArea(cnt) + area

    #Re-plot and show Step 6
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(image1)
    ax[0].set_title('Original image_rgb', fontsize=12)
    ax[0].set_axis_off()


    # Final Mask image_rgb
    ax[1].imshow(edges, cmap='gray')
    ax[1].set_title('Final Mask', fontsize=12)
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.show()

    #Return value to user Step 7

    return area/4
