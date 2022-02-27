"""
The below work was done by John David Maunder

"""

import cv2 as cv
from rgChromaticity import  mask_algo
from RGB import three_colour
from Canny import canny
from thresholding import global_threshold

if __name__ == '__main__':

	"""
	The code will start off by importing six photos from the bug data set.
	We will then crop(pre=process) and add these photo's to a list to make it easier
	to run our four algorithms.  
	"""
	image_list = []

	# import and crop first image
	image1 = cv.imread("IMG_2779.JPG")
	image1 = image1[327:621, 435:831]
	image_list.append(image1)


	# import and crop the second image
	image2 = cv.imread("IMG_2780.JPG")
	image2 = image2[328:654, 418:874]
	image_list.append(image2)

	# import and crop the third image
	image3 = cv.imread("IMG_2781.JPG")
	image3 = image3[210:531, 428:871]
	image_list.append(image3)

	# import and crop the fourth image
	image4 = cv.imread("IMG_2782.JPG")
	image4 = image4[306:621, 390:791]
	image_list.append(image4)

	# import and crop the fifth image
	image5 = cv.imread("IMG_2783.JPG")
	image5 = image5[268:543, 377:786]
	image_list.append(image5)

	# import and crop the fifth image
	image6 = cv.imread("IMG_2784.JPG")
	image6 = image6[298:635, 358:786]
	image_list.append(image6)
	pixel_values = []
	pixel_values2 = []
	pixel_values3 = []
	pixel_values4 = []


	"""
	The four algorithms below will run and return the expected pixel area of each image.
	Since there are six test images, each algorithm will return an array of six pixel are amounts
	"""

	#The first algorithm run will be the RGB algorithm
	for i in range(len(image_list)):
	   pixel_values.append(three_colour(image_list[i-1],4))

	print("Pixel Amounts for the RGB colour Algorithm:")
	print(pixel_values)


	#The Second Algorithm we will run is the RG Chromaticity Algorithm
	for i in range(len(image_list)):
		pixel_values2.append(mask_algo(image_list[i-1],4))

	print("Pixel Amounts for the RG Chromaticity Algorithm:")
	print(pixel_values2)

	# The Second Algorithm we will run is the RG Canny Algorithm
	for i in range(len(image_list)):
		pixel_values3.append(canny(image_list[i-1]))

	print("Pixel Amounts for the Canny Algorithm:")
	print(pixel_values3)

	#the fourth Algoirthm we will run is the Global Thresholding algorithm
	for i in range(len(image_list)):
		pixel_values4.append(global_threshold(image_list[i-1] , 4))

	print("Pixel Amounts for the global thresholding Algorithm:")
	print(pixel_values4)




