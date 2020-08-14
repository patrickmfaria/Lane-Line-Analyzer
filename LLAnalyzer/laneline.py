# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 07:04:06 2020

@author: Patrick de Faria
"""

import numpy as np
import cv2
from imagehandler import ImageHandler


# LINE CLASS for storing values
class Line:
    def __init__(self):
        # Was line detected in the previous frame?
        self.detected = False
        # Recent fits
        self.previous_fits = []
        # Polynomial coefficients averaged
        self.best_fit = None
        # X values plotted
        self.current_fitx = None
        # Polynomial coefficients
        self.current_fit = np.array([0, 0, 0])    # back to 0, 0, 0 -> False ?
        # Polynomial coefficients
        self.previous_fit = np.array([0, 0, 0])
        # Radius of curvature
        self.radius_of_curvature = 1000
        # Position of vehicle (dist from center)
        self.line_base_pos = None
        # Difference in coefficients
        self.diffs = np.array([0, 0, 0], dtype='float')
        # For first frame
        self.initialized = False
        # Average fit
        self.average_fit = np.array([0, 0, 0])
        # Average curvature
        self.average_curvature = 0
        # Previous curves
        self.previous_curves = []
        # Frame counter
        self.frame_cnt = 0

class Lane:
    def __init__(self):
        self.bottom_width = 0
        self.top_width = 0
        self.average_bottom_width = 0
        self.average_top_width = 0
        self.previous_bottom_widths = []
        self.previous_top_widths = []
        
class LanePipeline(object):
    def __init__(self, height, width, nwindows=18):
        self.height = height
        self.width = width
        self.nwindows = nwindows
        self._imagehandler = ImageHandler()
        self._nwindows = nwindows
        self._margim = 30
        self._minpix = 20
        
    def calcHistogram(self, transformed_egdes):
        self.histogram = np.sum(transformed_egdes[int(self.height // 2):, :], axis=0)
        return self.histogram
    
    def mask_region_of_interest(self,img):
        mask = np.zeros_like(img)
        unmasked_pixel_value = 1
        vertices = np.array([[(0, 720), (500, 450), (780, 450), (1280, 720)]], dtype=np.int32)
    
        cv2.fillPoly(mask, vertices, unmasked_pixel_value)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    
    def sliding_windows(self, img):
    
        # Calculate histogram by summing bottom half of image
        bottom_half = img[img.shape[0] // 2:, :]
        histogram = np.sum(bottom_half, axis=0)
    
        # Output image for testing
        out_img =np.zeros([img.shape[0], img.shape[1], 3],dtype=np.uint8)
        #out_img = np.dstack((img, img, img))
    
        # Create starting point on left and right side and set them as current points
        midpoint = np.int(histogram.shape[0] // 2)
                
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        leftx_base = np.argmax(histogram[:midpoint])

        leftx_current = leftx_base
        rightx_current = rightx_base
    
        # Parameters of sliding window
        # Number of windows
        nwindows = self.nwindows
        # Width of windows
        margin = self._margim
        # Minimum number of pixels to recenter window
        minpix = self._minpix
    
        # Set window height
        window_height = np.int(img.shape[0] // nwindows)
    
        # Find nonzero pixels
        nonzero = img.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])
    
        # Empty lists for storing lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
    
        # Step through windows
        for window in range(nwindows):
    
            # Window boundaries:
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
    
            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
    
            # Identify pixels within the windows
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    
            # Add the found pixels to lane line
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
    
            # Update x axis position based on pixels found
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
        # Concatenate list of pixels
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Get left and right lane pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Color left and right line pixels
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]
    
        # Fit polynomial based on pixels found
        left_fit, right_fit = self.fit_poly(leftx, lefty, rightx, righty)
    
        # # For debugging
        left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
        right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])
    
        # Add text to debugging image
        cv2.putText(out_img, left_fit_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
        cv2.putText(out_img, right_fit_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
    
        return left_fit, right_fit, out_img

    def search_around_poly(self, binary_warped, pleft_fit, pright_fit):
    
        # Margin for searching around curve
        margin = self._margim
    
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
    
        # Mask to get non-zero pixels that are next to the curve within margin
        left_lane_inds = ((nonzerox > (pleft_fit[0] * (nonzeroy ** 2) + pleft_fit[1] * nonzeroy +
                                        pleft_fit[2] - margin)) & (nonzerox < (pleft_fit[0] * (nonzeroy ** 2) +
                                        pleft_fit[1] * nonzeroy + pleft_fit[2] + margin)))
    
        right_lane_inds = ((nonzerox > (pright_fit[0] * (nonzeroy ** 2) + pright_fit[1] * nonzeroy +
                                        pright_fit[2] - margin)) & (nonzerox < (pright_fit[0] * (nonzeroy ** 2) +
                                        pright_fit[1] * nonzeroy + pright_fit[2] + margin)))
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    
        # Generate y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    
        try:
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
        except :
            left_fit = pleft_fit
            right_fit = pright_fit    
            
        left_fitx, right_fitx = self.calc_x_values(binary_warped.shape, left_fit, right_fit)
    
        # Create an image to draw on and an image to show the selection window
        out_img =np.zeros([binary_warped.shape[0], binary_warped.shape[1], 3],dtype=np.uint8)

        window_img = np.zeros_like(out_img)
    
        # Color left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
    
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
        # For debugging
        left_fit_text = "left: %.6f %.6f %.6f" % (left_fit[0], left_fit[1], left_fit[2])
        right_fit_text = "right: %.6f %.6f %.6f" % (right_fit[0], right_fit[1], right_fit[2])
    
        # Add text to debugging image
        cv2.putText(out_img, left_fit_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
        cv2.putText(out_img, right_fit_text, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=2)
    
        return left_fit, right_fit, out_img
    
    def fit_poly(self, leftx, lefty, rightx, righty):
    
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def calc_x_values(self, img_shape, left_fit, right_fit):
        # Generate y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    
        # Calculate x values using polynomial coeffs
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
        return left_fitx, right_fitx

    def sanity_check(self, left_lane, right_lane, lane):
    
        # Calculate widths at top and bottom
        top_width_diff = abs(lane.top_width - lane.average_top_width)
        bottom_width_diff = abs(lane.bottom_width - lane.average_bottom_width)

        # Define sanity checks      
        width_check_top = (top_width_diff > 0.2 * lane.average_top_width or lane.top_width > 1.25 * lane.bottom_width) or (lane.top_width <=0 or lane.bottom_width <= 0 )
        width_check_bottom = bottom_width_diff > 0.05 * lane.average_bottom_width
        lane_intersect_check = lane.top_width < 0.0 or lane.bottom_width < 0.0
        curve_check = right_lane.current_fit[0] * left_lane.current_fit[0] < -0.00005 * 0.0001
    
        # Check if parameters are ok (skip for first frame)
        if (left_lane.frame_cnt > 1) and (right_lane.frame_cnt > 1):
    
            if width_check_bottom:
                result = False
    
            elif width_check_top:
                result = False
    
            elif lane_intersect_check:
                result = False
    
            elif curve_check:
                result = False
    
            else:
                result = True
    
        else:
            result = True
    
        return result
    
    def average_fits(self,img_shape, lane):
        n = 5
        average_fit = [0, 0, 0]
    
        # If we do not have enough fits, append the list with the current fit
        if len(lane.previous_fits) < n:
            lane.previous_fits.append(lane.current_fit)
        # If amount of fits == n, remove the last element and add the current one
        if len(lane.previous_fits) == n:
            lane.previous_fits.pop(0)     #(n-1)
            lane.previous_fits.append(lane.current_fit)           #insert(0, lane.current_fit)
    
        # If we have enough fits, calculate the average
        if len(lane.previous_fits) > 0:
            for i in range(0, 3): # fro each element of the fits 
                total = 0
                for num in range(0, len(lane.previous_fits)):
    
                    total = total + lane.previous_fits[num][i]
    
                average_fit[i] = total / len(lane.previous_fits)
    
        return average_fit
    
    def average_width(self, img_shape, lane):
        sum_bottom = 0
        sum_top = 0
        n = 3
        average_bottom_width = 0
        average_top_width = 0
    
        if len(lane.previous_bottom_widths) < n:
            lane.previous_bottom_widths.append(lane.bottom_width)
        # If amount of fits == n, remove the last element and add the current one
        if len(lane.previous_bottom_widths) == n:
            lane.previous_bottom_widths.pop(n-1)
            lane.previous_bottom_widths.insert(0, lane.bottom_width)
    
        # If we have enough fits, calculate the average
        if (len(lane.previous_bottom_widths) > 0):
            for i in range(0, len(lane.previous_bottom_widths)):
    
                sum_bottom = sum_bottom + lane.previous_bottom_widths[i]
                average_bottom_width = sum_bottom / len(lane.previous_bottom_widths)
    
    
        if len(lane.previous_top_widths) < n:
            lane.previous_top_widths.append(lane.top_width)
        # If amount of fits == n, remove the last element and add the current one
        if len(lane.previous_top_widths) == n:
            lane.previous_top_widths.pop(n-1)
            lane.previous_top_widths.insert(0, lane.top_width)
    
        # If we have enough fits, calculate the average
        if (len(lane.previous_top_widths) > 0):
            for i in range(0, len(lane.previous_top_widths)):
    
                sum_top = sum_top + lane.previous_top_widths[i]
                average_top_width = sum_top / len(lane.previous_top_widths)
    
        return average_bottom_width, average_top_width
       
    def measure_curvature_real(self, img_shape, fit):
        # Calculates the curvature of polynomial functions in meters
        ym_per_pix = 30 / 310  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 300  # meters per pixel in x dimension
    
        # Generate y values for plotting
        ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    
        # Calculate x values using polynomial coeffs
        fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    
        # Evaluate at bottom of image
        y_eval = np.max(ploty)
    
        # Fit curves with corrected axes
        curve_fit = np.polyfit(ploty * ym_per_pix, fitx * xm_per_pix, 2)
    
        # Calculate curvature values for left and right lanes
        curvature = ((1 + (2 * curve_fit[0] * y_eval * ym_per_pix + curve_fit[1]) ** 2) ** 1.5) / np.absolute(
            2 * curve_fit[0])
    
        return curvature
    
    def get_vehicle_position(self, img_shape, left_lane_pos, right_lane_pos):
        xm_per_pix = 3.7 / 300  # meters per pixel in x dimension
    
        # Calculate position based on midpoint - center of lanes distance
        midpoint = img_shape // 2
        center_of_lanes = (right_lane_pos + left_lane_pos) / 2
        position = midpoint - center_of_lanes
        
        # Get value in meters
        real_position = position * xm_per_pix
   
        return real_position

    def draw_lanes(self, warped, undist, left_fit, right_fit, curvature, Minv, frame, st,  frames_per_second):
    
        # Generate y values
        ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    
        # Calculate x values
        left_fitx, right_fitx = self.calc_x_values(warped.shape, left_fit, right_fit)
        xm_per_pix = 3.7 / 300  # meters per pixel in x dimension
    
        # Calculate position based on midpoint - center of lanes distance
        midpoint = undist.shape[1] // 2
        center_of_lanes = (right_fitx[undist.shape[0]-1] + left_fitx[undist.shape[0]-1]) / 2
        position = midpoint - center_of_lanes
        
        # Get value in meters
        real_position = position * xm_per_pix

        # Create image to draw lines onto
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # plot midpoint
    
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
    
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    
        # Combine the result with the original image
        lanes = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
        summary = np.dstack((warp_zero, warp_zero, warp_zero))
        # # Texts to write on image
        curv_text = "Curvature: %.2f meters" % curvature
        
        if real_position >= 0:
            towards = 'left'
        else:
            towards = 'right'
        
        pos_text = 'Vehicle is {0:.1f}m '.format(abs(real_position)) + towards +' of center.'
        
        frame_position = "frame counter: %2.2f" % frame
        sanity_test_text = "Sanity Test: " + ("Passed" if st else "Not Passed")
        frames_per_second_text = "frames_per_second: " + str(frames_per_second)    

        # # Add text to image
        cv2.putText(summary, curv_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
        cv2.putText(summary, pos_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 200))
        cv2.putText(summary, frame_position, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255))
        cv2.putText(summary, sanity_test_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255))
        cv2.putText(summary, frames_per_second_text, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 250, 55))

        return lanes,summary
     
 
