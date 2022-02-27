import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
"""
1.	The original image is cropped so that the four-leaf objects become the focal point of the image
2.	The cropped image is then converted to greyscale
3.	The image is pre-processed using a blurring algorithm
4.	A patch of pixels is chosen from a leaf object. The high and low values from this patch are used as the thresholds. To pass the threshold, the pixel being tested must fall between all three colour channel values. 
5.	A nested for loop, goes through the greyscale image and converts pixels that are within the threshold to white and the other pixels to black. 
6.	The black and white pixels are replotted and shown next to the original image.
7.	The total average area of the leaves is calculated by summing up the number of white pixels in the new black and white image. 

The two parameters are the cropped image passed by the user (step 1) and the leaf_count which is usually four
"""

def three_colour (img,leaf_count):
    #Step 2 convert to grayscale and this image will be our ending mask
    image_binary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #pre-processing by dilating and blurrring to remove shadows (Step 3)
    dilated_img = cv.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv.medianBlur(dilated_img, 21)
    diff_img = 255 - cv.absdiff(img, bg_img)

    #Convert to RGB since we need this to compare to our threshold
    image_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #used for iterations
    x = image_rgb.shape[0]
    y = image_rgb.shape[1]

    #step 5
    for i in range(x):
        for j in range(y):
            if (image_rgb[i][j][0] > 60 and image_rgb[i][j][0] < 100) and (
                    image_rgb[i][j][1] > 60 and image_rgb[i][j][1] < 105) and (
                    image_rgb[i][j][2] > 40 and image_rgb[i][j][2] < 85):
                image_binary[i][j] = 255
            else:
                image_binary[i][j] = 0

    count =0
    for i in range(x):
        for j in range(y):
            if image_binary[i][j] > 0:
                count+=1

    #Step 6 plotting new image
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(img)
    ax[0].set_title('Original image_rgb', fontsize=12)
    ax[0].set_axis_off()

    # Final Mask image_rgb
    ax[1].imshow(image_binary, cmap='gray')
    ax[1].set_title('Final Mask', fontsize=12)
    ax[1].set_axis_off()
    fig.tight_layout()
    plt.imshow(image_binary, cmap='gray')
    plt.show()

    #Step 7 return value to user
    average_leaf_pixel_size = count/leaf_count

    return average_leaf_pixel_size