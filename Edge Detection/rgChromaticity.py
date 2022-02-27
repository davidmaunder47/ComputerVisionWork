# basic function to get the guassian number per patch pixel.
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
"""
1.	The original image is cropped so that the four-leaf objects become the focal point of the image
2.	The cropped image is then converted to greyscale
3.	The “patch” is taken from one green area on a leaf object. The patch is used for each testing image
4.	The RG ratio for the whole image is compared to the RG ratio from the patch threshold 
5.	A nested for loop, goes through the greyscale image and converts pixels that are within the threshold to white and the other pixels to black. A pixel is converted to white if the RG ratio is within 3% of the patch ratio.
6.	The black and white pixels are replotted and shown next to the original image.
7.	The total average area of the leaves is calculated by summing up the number of white pixels in the new black and white image. 

The two parameters are the cropped image passed by the user (step 1) and the leaf_count which is usually four
"""

#used as an intermediate function to calculate the gaussian for a given patch
def gaussian(p, mean, std):
    return np.exp(-(p - mean) ** 2 / (2 * std ** 2)) * (1 / (std * ((2 * np.pi) ** 0.5)))


def mask_algo(image, leaf_count, mean=1, std=1):
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # convert the original image_rgb to its red and green components per each pixel
    image_rgb_r = image_rgb[:, :, 0] / image_rgb.sum(axis=2)
    image_rgb_g = image_rgb[:, :, 1] / image_rgb.sum(axis=2)

    # This is the patch I choose for this algorithm
    #The same patch will be used for all six images
    base_image = cv.imread("IMG_2779.JPG")
    base_image_rgb = cv.cvtColor(base_image, cv.COLOR_BGR2RGB)
    patch = base_image_rgb[400:410,530:540]

    plt.imshow(patch)
    plt.show()

    patch_r = patch[:, :, 0] / patch.sum(axis=2)
    patch_g = patch[:, :, 1] / patch.sum(axis=2)

    # Calculate the standard deviation and the mean for the patches
    std_patch_r = np.std(patch_r.flatten())
    mean_patch_r = np.mean(patch_r.flatten())

    std_patch_g = np.std(patch_g.flatten())
    mean_patch_g = np.mean(patch_g.flatten())

    # this is used for our input to display the red and green masks
    # this is to see the different red and green components
    masked_image_rgb_r = gaussian(image_rgb_r, mean_patch_r, std_patch_r)
    masked_image_rgb_g = gaussian(image_rgb_g, mean_patch_g, std_patch_g)

    # now calculate the guassian numbers from the formulas above
    guass_patch_r = gaussian(patch_r, mean_patch_r, std_patch_r)
    guass_patch_g = gaussian(patch_g, mean_patch_g, std_patch_g)

    # take the red/green image_rgb for each patch pixel
    patch_ratio = (patch_r / patch_g)
    # take the average of the patch since we just want one number for our threshold
    # our threshold ratio
    final_patch_ratio = (patch_ratio.sum() / (patch_ratio.size))

    # this will convert the original image_rgb into its red and green components
    # we will then compare this to our threshold in our for loop below
    final_mask = (image_rgb_r / image_rgb_g)

    # this will yeild us our black and white output
    x = final_mask.shape[0]
    y = final_mask.shape[1]
    for i in range(x):
        for j in range(y):
            if (final_mask[i][j] > final_patch_ratio * 1.03) or (final_mask[i][j] < final_patch_ratio * 0.97):
                final_mask[i][j] = 0

    # the rest of the code is used to display the various image_rgbs created above

    # Original image_rgb

    fig, ax = plt.subplots(1, 4, figsize=(10, 6))
    ax[0].imshow(image_rgb)
    ax[0].set_title('Original image_rgb with Patch', fontsize=10)
    ax[0].set_axis_off()

    # Final Mask image_rgb
    ax[1].imshow(final_mask, cmap= 'gray');
    ax[1].set_title('Final Mask', fontsize=22)
    ax[1].set_axis_off()
    fig.tight_layout()

    # Green Mask image_rgb
    ax[2].imshow(masked_image_rgb_g);
    ax[2].set_title('Mask_green', fontsize=22)
    ax[2].set_axis_off()
    fig.tight_layout()

    # Red Maks image_rgb
    ax[3].imshow(masked_image_rgb_r, cmap ="gray");
    ax[3].set_title('Mask_red', fontsize=22)
    ax[3].set_axis_off()
    fig.tight_layout()
    plt.show()

    #Count each white pixel amount
    count = 0
    for i in range(x):
        for j in range(y):
            if final_mask[i][j] > 0:
                count+=1

    #Return value to user
    return count/leaf_count



