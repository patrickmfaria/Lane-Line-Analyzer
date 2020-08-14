# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 06:09:12 2020

@author: Patrick de Faria
"""

import cv2
import numpy as np

class ImageHandler:
    
    def __init__(self):
        
        self.α = 0.9                 # Weight factor for initial image
        self.β = 0.4                 # Weight factor for new image
        self.λ = 0.22                # Scalar added to each sum (of new and initial image), see weighted_img function
        self.kernel_size = 7         # Size of the n x n matrix used as kernel in gaussian blur
        self.low_threshold = 50      # Value for the canny function, defining the first threshold
        self.high_threshold = 150    # Value for the canny function, defining the second threshold
        
        self.ly = np.array([20, 100, 100], dtype = "uint8") # Low lalue for yellow HSV.
        self.uy = np.array([30, 255, 255], dtype = "uint8") # Hig value for yellow HSV.

  
    def perspective_transform(self, image, src, dst):
        height = image.shape[0]
        width = image.shape[1]
        
        # Given src and dst points we calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image
        warped = cv2.warpPerspective(image, M, (width, height))
        # We also calculate the oposite transform
        unwrap_m = cv2.getPerspectiveTransform(dst, src) #minv
        unwarped = cv2.warpPerspective(image, unwrap_m, (width, height))
        # Return the resulting image and matrix 
        return (warped, M, unwrap_m, unwarped)
    
    def resize_scale(self, img, scale_percent=0.50):
        width   = int(img.shape[1] * scale_percent)
        height  = int(img.shape[0] * scale_percent)
        dimension = (width,height)
        
        resized = cv2.resize(img, dimension, interpolation = cv2.INTER_AREA)
        return resized
        
    def adjust_gamma(self, image, gamma=1.0):
    	# build a lookup table mapping the pixel values [0, 255] to
    	# their adjusted gamma values
    	invGamma = 1.0 / gamma
    	table = np.array([((i / 255.0) ** invGamma) * 255
    		for i in np.arange(0, 256)]).astype("uint8")
    	# apply gamma correction using the lookup table
    	return cv2.LUT(image, table)
    
    def remove_black(self, img):
        
        height = img.shape[0]
        width = img.shape[1]        

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray,3)
        
        (_, mask) = cv2.threshold(gray, 1.0, 255.0, cv2.THRESH_BINARY);

        # findContours destroys input
        temp = mask.copy()
        (contours, _) = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours by largest first (if there are more than one)
        contours = sorted(contours, key=lambda contour:len(contour), reverse=True)
        if contours:
            roi = cv2.boundingRect(contours[0])

            # use the roi to select into the original 'stitched' image
            img2 = img[roi[1]:roi[3], roi[0]:roi[2]]
            return cv2.resize(img2, (width, height), interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            
        
    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        aimga = np.array(imageA)
        aimgb = np.array(imageB)
        
        print(aimga.shape)
        print(aimgb.shape)
        
        err = np.sum((aimga.astype("float") - aimgb.astype("float")) ** 2)
        err /= float(aimga.shape[0] * aimgb.shape[1])
        
        return err
    
        # return the MSE, the lower the error, the more "similar"
        # the two images are
    
    def grayscale(self, img):
        """Applies the Grayscale transform"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def canny(self, img, low_threshold, high_threshold):
        """Applies the Canny transform, which tracks edges by hysteresis"""
        return cv2.Canny(img, low_threshold, high_threshold)
    
    def gaussian_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def weighted_img(self, img, initial_img, α, β, λ):
        """Calculating the weighted sum of the two image arrays"""
        # (initial_img * α) + (img * β) + λ
        return cv2.addWeighted(initial_img, α, img, β, λ)
