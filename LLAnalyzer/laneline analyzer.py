# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 21:42:51 2020

@author: Patrick de Faria
"""

import cv2
import numpy as np
from camera import Camera
from canvas import  CanvasObj, Canvas
from imagehandler import ImageHandler
from laneline import LanePipeline, Line, Lane
import glob
import datetime as dt
            
def nothing(*arg):
    pass

# Inout Video
file_path  = "videos/20200622_141857.mp4"

# Get the calibration Matrix
calibrated_camera = Camera(glob.glob('/camera_cal/calibration*.jpg'))

video_reader =  cv2.VideoCapture(file_path)

frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
total_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        
success = 1
firsttime= True
calibrated = None
elapsedtime = 0.0

# You can use it to jump to a specific Frame
jump = 500

# canvas control
#
#
# define the initial postion of the main screen
#
initial_leftpos = 0
initial_toppos =  0

# definie the dimensions of the imagens
# here as more larger more slow
height = 216
width = 384

# Create the canvas object
canvas = Canvas(objheight = height, objwidth = width)
imagehandler = ImageHandler()

nameApp = "LaneLine Analyzer"
nameAppControlPanel = "LLA Control Panel"

cv2.namedWindow(nameApp)
cv2.namedWindow(nameAppControlPanel)
cv2.moveWindow(nameApp, initial_leftpos, initial_toppos);

# Set the adjust controls range values
cv2.createTrackbar('b_thresh_min', nameAppControlPanel, 1, 255, nothing)
cv2.createTrackbar('b_thresh_max', nameAppControlPanel, 1, 255, nothing)
cv2.createTrackbar('l_perc_min', nameAppControlPanel, 1, 100, nothing)
cv2.createTrackbar('l_perc_max', nameAppControlPanel, 1, 100, nothing)
cv2.createTrackbar('sx_perc_min', nameAppControlPanel, 1, 100, nothing)
cv2.createTrackbar('sx_perc_max', nameAppControlPanel, 1, 100, nothing)

cv2.createTrackbar('n_windows', nameAppControlPanel, 1, 50, nothing)
cv2.createTrackbar('margin_window', nameAppControlPanel, 1, 100, nothing)
cv2.createTrackbar('min_median', nameAppControlPanel, 1, 100, nothing)

cv2.createTrackbar('Perspective_Factor_Width', nameAppControlPanel, 1, 100, nothing)
cv2.createTrackbar('Perspective_Height', nameAppControlPanel, 1, 100, nothing)
cv2.createTrackbar('Perspective_Bottom_Margim', nameAppControlPanel, 0, 100, nothing)
cv2.createTrackbar('Perspective_Bottom_Width', nameAppControlPanel, 0, 200, nothing)
cv2.createTrackbar('Perspective_Destination_Factor', nameAppControlPanel, 1, 200, nothing)

# Set the adjust controls default values
cv2.setTrackbarPos('b_thresh_min', nameAppControlPanel, 125)
cv2.setTrackbarPos('b_thresh_max', nameAppControlPanel, 200)
cv2.setTrackbarPos('l_perc_min', nameAppControlPanel, 70)
cv2.setTrackbarPos('l_perc_max', nameAppControlPanel, 100)
cv2.setTrackbarPos('sx_perc_min', nameAppControlPanel, 90)
cv2.setTrackbarPos('sx_perc_max', nameAppControlPanel, 100)

cv2.setTrackbarPos('n_windows', nameAppControlPanel, 18)
cv2.setTrackbarPos('margin_window', nameAppControlPanel, 18)
cv2.setTrackbarPos('min_median', nameAppControlPanel, 40)

cv2.setTrackbarPos('Perspective_Factor_Width', nameAppControlPanel, 70)
cv2.setTrackbarPos('Perspective_Height', nameAppControlPanel, 65)
cv2.setTrackbarPos('Perspective_Bottom_Margim', nameAppControlPanel, 10)
cv2.setTrackbarPos('Perspective_Bottom_Width', nameAppControlPanel, 50)
cv2.setTrackbarPos('Perspective_Destination_Factor', nameAppControlPanel, 95)

# Variables for tjhe DVR control
paused = False
frame_position = 0
rewind = False
forward = False
start_count = True
start_time = 0
count24_frame = 0
frames_per_second = 0

#
# Laneslines control
#
#

lanepipeline = LanePipeline(height =height, width=width)
left_lane = Line()
right_lane = Line()
lane = Lane()

cap = cv2.VideoCapture(0)

while success:
    
    #
    # Video Control
    # <Space> Pause, R - Rewind, F - Forward
    #
    if jump != 0:
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, jump) #Jump to the last one frame
        jump = 0

    # if video is not paused keep pulling frames
    if not paused:
        success, image = video_reader.read()
        #success, image = cap.read() ### you can swich to the camera
        buffer = image
        frame_position =  int(video_reader.get(cv2.CAP_PROP_POS_FRAMES))

        # if finish the video just start it over
        if not success:
            frame_position = 0
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position) #Jump to the last one frame
            success, image = video_reader.read()
            buffer = image
            frame_position =  int(video_reader.get(cv2.CAP_PROP_POS_FRAMES))

    # you can only uyse rewind and forward if paused
    else:
        if rewind:
            rewind = False
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position) #Jump to the last one frame
            success, image = video_reader.read()
            buffer = image
        else:
            image = buffer
        if forward:
            forward = False
            video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_position) #Jump to the last one frame
            success, image = video_reader.read()
            buffer = image
        else:
            image = buffer

    # if read the image then process it
    if(success):
        
        #
        # Gests the Trackbar values all the time
        # 
        #
        b_thresh_min = cv2.getTrackbarPos('b_thresh_min', nameAppControlPanel)
        b_thresh_max = cv2.getTrackbarPos('b_thresh_max', nameAppControlPanel)
        l_perc_min = cv2.getTrackbarPos('l_perc_min', nameAppControlPanel)
        l_perc_max = cv2.getTrackbarPos('l_perc_max', nameAppControlPanel)
        sx_perc_min = cv2.getTrackbarPos('sx_perc_min', nameAppControlPanel)
        sx_perc_max = cv2.getTrackbarPos('sx_perc_max', nameAppControlPanel)

        pfw = cv2.getTrackbarPos('Perspective_Factor_Width', nameAppControlPanel)
        pph = cv2.getTrackbarPos('Perspective_Height', nameAppControlPanel)
        pbm = cv2.getTrackbarPos('Perspective_Bottom_Margim', nameAppControlPanel)
        pbw = cv2.getTrackbarPos('Perspective_Bottom_Width', nameAppControlPanel)
        pdf = cv2.getTrackbarPos('Perspective_Destination_Factor', nameAppControlPanel)
        
        b_thresh=(b_thresh_min, b_thresh_max)
        l_perc=(l_perc_min, l_perc_max)
        sx_perc=(sx_perc_min, sx_perc_max)

        n_windows= cv2.getTrackbarPos('n_windows', nameAppControlPanel)
        margim = cv2.getTrackbarPos('margin_window', nameAppControlPanel)
        minpix = cv2.getTrackbarPos('min_median', nameAppControlPanel)
                                
        if start_count:
            start_time = dt.datetime.today().timestamp()
            start_count = False
         
        #
        #
        # Start the Image Pipeline
        #
        #
        #
        #image = resize_scale(image, scale_percent=scale_percent)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

        # undistorted image 
        image = calibrated_camera(image)
        (height, width, _) = image.shape

        canvas.addObj(CanvasObj('Original', 1, 1, image))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

        b_channel = lab[:, :, 2]
        # Threshold b color channel
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1
        b_binary_overlay = np.dstack((b_binary, b_binary, b_binary)) * 255
        canvas.addObj(CanvasObj('Threshold Lab color channel', 1, 2, lab))

        # Convert to LUV color space
        luv = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
        l_channel = luv[:, :, 0]

        # Create percentile-based thresholds
        l_thresh_min = np.percentile(l_channel, l_perc[0])
        l_thresh_max = np.percentile(l_channel, l_perc[1])

        # Threshold l color channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        l_binary_overlay = np.dstack((l_binary, l_binary, l_binary)) * 255
        canvas.addObj(CanvasObj('Threshold lUV color channel', 1, 3, l_binary_overlay))

        # Find edges with Sobelx
        sobel_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobel_x)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        scaled_sobel_overlay = np.dstack((scaled_sobel, scaled_sobel, scaled_sobel)) #* 255
        canvas.addObj(CanvasObj('Scaled Sobel', 1, 4, scaled_sobel_overlay))
        
        # Create percentile-based thresholds
        sx_thresh_min = np.percentile(scaled_sobel, sx_perc[0])
        sx_thresh_max = np.percentile(scaled_sobel, sx_perc[1])

        # Threshold edges (x gradient)
        sx_binary = np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel >= sx_thresh_min) & (scaled_sobel <= sx_thresh_max)] = 1
        sx_binary_overlay = np.dstack((sx_binary, sx_binary, sx_binary)) * 255
        canvas.addObj(CanvasObj('Scaled Sobel Threshold', 2, 1,sx_binary_overlay))

        # Get white edges
        sobel_white_binary = np.zeros_like(l_channel)
        sobel_white_binary[(sx_binary == 1) & (l_binary == 1)] = 1
    
        # Get yellow edges
        sobel_yellow_binary = np.zeros_like(l_channel)
        sobel_yellow_binary[(sx_binary == 1) & (b_binary == 1)] = 1

        # Output image for debugging
        white_sobelx_and_color = np.dstack((sobel_white_binary, sobel_yellow_binary, np.zeros_like(sobel_white_binary))) * 255
        canvas.addObj(CanvasObj('white sobelx and color', 2, 2, white_sobelx_and_color))

        # Output image for pipeline
        combined_binary_sobel = np.zeros_like(b_binary)
        combined_binary_sobel[((sobel_white_binary == 1) | (sobel_yellow_binary == 1) ) ] = 1
        #combined_binary_sobel[((sobel_white_binary == 1) | (sobel_yellow_binary == 1) ) | (b_binary == 0)] = 1
        combined_binary_sobel_overlay = np.dstack((combined_binary_sobel, combined_binary_sobel, combined_binary_sobel)) * 255

        # 3) Apply perspective transform
        s1 = [width // 2 - pfw, height * (pph/100)]
        s2 = [width // 2 + pfw, height * (pph/100)]
        s3 = [-pbw, height - pbm]
        s4 = [width + pbw, height - pbm]
        src = np.float32([s1, s2, s3, s4])

        cv2.polylines(combined_binary_sobel_overlay, np.int32([[s1,s2,s4,s3]]), True, 255, 2)
                
        # Quadrangle verties coordinates in the destination image
        d1 = [pdf, 0]
        d2 = [width - pdf, 0]
        d3 = [pdf, height]
        d4 = [width - pdf, height]
        dst = np.float32([d1, d2, d3, d4])
        
        canvas.addObj(CanvasObj('Combined binary sobel', 2, 3 ,combined_binary_sobel_overlay))
        top_view, M, Minv, unwarped =  imagehandler.perspective_transform(combined_binary_sobel, src, dst)

        # calculate histogram
        h = np.zeros((height,width,3), dtype=np.uint8)
        histogram = lanepipeline.calcHistogram (top_view)
        for x,y in enumerate(histogram):
            cv2.line( h,(int(x),0),(int(x),int(y)),(205,150,55))
        h = np.flipud(h)
        
        h2 = h[:, :, 0] * 255
        h3 = cv2.addWeighted(top_view, 1,  h2, 1.9, 0)
        unwarped_overlay = np.dstack((unwarped, unwarped, unwarped)) * 255
        canvas.addObj(CanvasObj('Inverse Perspective', 3, 1 ,unwarped_overlay))
        canvas.addObj(CanvasObj('Threshold Lab color channel', 3, 2 ,b_binary_overlay))

        #
        #
        #
        #  Start lane detection
        #
        #
        lanepipeline.nwindows = n_windows
        lanepipeline._margim = margim
        lanepipeline._minpix = minpix
        
        # 4) Detect lane pixels and fit polynomial
        # If previous lane was detected, search next to curve, otherwise use sliding window method
        if (left_lane.detected is False) or (right_lane.detected is False):
            try:
                    left_fit, right_fit, lanes_colored = lanepipeline.sliding_windows(top_view)
            # if nothing was found, use previous fit
            except TypeError:
                left_fit = left_lane.previous_fit
                right_fit = right_lane.previous_fit
                lanes_colored = np.zeros_like(image)
        else:
            try:
                left_fit, right_fit, lanes_colored = lanepipeline.search_around_poly(top_view, 
                                                                                     left_lane.previous_fit, 
                                                                                     right_lane.previous_fit)
            except TypeError:
                try:
                    left_fit, right_fit, lanes_colored = lanepipeline.sliding_windows(top_view)
                # if nothing was found, use previous fit
                except TypeError:
                    left_fit = left_lane.previous_fit
                    right_fit = right_lane.previous_fit
                    lanes_colored = np.zeros_like(image)
    
        canvas.addObj(CanvasObj('Lanes Colored', 3, 3, lanes_colored ))
    
        left_lane.current_fit = left_fit
        right_lane.current_fit = right_fit
    
        # Calculate base position of lane lines to get lane distance
        # calculate the X using height as Y
        left_lane.line_base_pos = left_fit[0] * (top_view.shape[0] - 1) ** 2 + left_fit[1] * (top_view.shape[0] - 1) + left_fit[2]
        right_lane.line_base_pos = right_fit[0] * (top_view.shape[0] - 1) ** 2 + right_fit[1] * (top_view.shape[0] - 1) + right_fit[2]
        left_lane.line_mid_pos = left_fit[0] * (top_view.shape[0] // 2) ** 2 + left_fit[1] * (top_view.shape[0] // 2) + left_fit[2]
        right_lane.line_mid_pos = right_fit[0] * (top_view.shape[0] // 2) ** 2 + right_fit[1] * (top_view.shape[0] // 2) + right_fit[2]
    
        # Calculate top and bottom position of lane lines for sanity check
        lane.top_width = right_fit[2] - left_fit[2]
        lane.bottom_width = right_lane.line_base_pos - left_lane.line_base_pos
        lane.middle_width = right_lane.line_mid_pos - left_lane.line_mid_pos

        top_view_overlay = np.dstack((h3, h3, h3)) * 255
        canvas.addObj(CanvasObj('Perspective + Lane', 2, 4 ,top_view_overlay))
                        
        # Check if values make sense
        sanity_check = lanepipeline.sanity_check(left_lane, right_lane, lane)
        if sanity_check is False:
            # If fit is not good, use previous values and indicate that lanes were not found
            if len(left_lane.previous_fits) > 0: #### 5:
                left_lane.current_fit = left_lane.average_fit
                right_lane.current_fit = right_lane.average_fit
            else:
                left_lane.current_fit = left_lane.previous_fit
                right_lane.current_fit = right_lane.previous_fit

            left_lane.detected = False
            right_lane.detected = False    
        else:
            # If fit is good, use current values and indicate that lanes were found
            left_lane.detected = True
            right_lane.detected = True
            
            left_lane.frame_cnt += 1
            right_lane.frame_cnt += 1

        # Calculate the average of the recent fits and set this as the current fit
        left_lane.average_fit = lanepipeline.average_fits(top_view.shape, left_lane)
        right_lane.average_fit = lanepipeline.average_fits(top_view.shape, right_lane)

        lane.average_bottom_width, lane.average_top_width = lanepipeline.average_width(top_view.shape, lane)

        # Calculate base position of lane lines to get lane distance
        # calculate the X using height as Y
        left_lane.line_base_pos = left_lane.current_fit[0] * (top_view.shape[0] - 1) ** 2 + left_lane.current_fit[1] * (top_view.shape[0] - 1) + left_lane.current_fit[2]
        right_lane.line_base_pos = right_lane.current_fit[0] * (top_view.shape[0] - 1) ** 2 + right_lane.current_fit[1] * (top_view.shape[0] - 1) + right_fit[2]
        left_lane.line_mid_pos = left_lane.current_fit[0] * (top_view.shape[0] // 2) ** 2 + left_fit[1] * (top_view.shape[0] // 2) + left_lane.current_fit[2]
        right_lane.line_mid_pos = right_lane.current_fit[0] * (top_view.shape[0] // 2) ** 2 + right_lane.current_fit[1] * (top_view.shape[0] // 2) + right_lane.current_fit[2]
    
        # Calculate top and bottom position of lane lines for sanity check
        lane.top_width = right_lane.current_fit[2] - left_lane.current_fit[2]
        lane.bottom_width = right_lane.line_base_pos - left_lane.line_base_pos
        lane.middle_width = right_lane.line_mid_pos - left_lane.line_mid_pos
                
        # Calculate position based on midpoint - center of lanes distance
        midpoint = top_view.shape[1] // 2

        cv2.line(top_view_overlay, (midpoint, 0 ), (midpoint,  top_view.shape[0]), (20, 100, 200), 2)    

        leftxtop = left_lane.current_fit[0] * 1 ** 2 + left_lane.current_fit[1] * 1 + left_lane.current_fit[2]
        leftxbottom = left_lane.current_fit[0] * top_view.shape[0] ** 2 + left_lane.current_fit[1] * top_view.shape[0] + left_lane.current_fit[2]
   
        rightxtop = right_lane.current_fit[0] * 1 ** 2 + right_lane.current_fit[1] * 1 + right_lane.current_fit[2]
        rightxbottom = right_lane.current_fit[0] * top_view.shape[0] ** 2 + right_lane.current_fit[1] * top_view.shape[0] + right_lane.current_fit[2]

        center_of_lanes_bottom = leftxbottom + ((rightxbottom - leftxbottom) / 2)
        center_of_lanes_top = leftxtop + ((rightxtop - leftxtop) / 2)

        cv2.line(top_view_overlay, (int(leftxtop), 1 ), (int(leftxbottom), top_view.shape[0]), (220, 25, 150), 1)    
        cv2.line(top_view_overlay, (int(rightxtop), 1 ), (int(rightxbottom), top_view.shape[0]), (220, 25, 150), 1)    
        cv2.line(top_view_overlay, (int(center_of_lanes_top), 1 ), (int(center_of_lanes_bottom), top_view.shape[0]), (120, 200, 100), 1)    

        # 5) Determine lane curvature and position of the vehicle
        left_lane.radius_of_curvature = lanepipeline.measure_curvature_real(top_view.shape, left_fit)
        right_lane.radius_of_curvature = lanepipeline.measure_curvature_real(top_view.shape, right_fit)
        curvature = left_lane.radius_of_curvature + right_lane.radius_of_curvature / 2
           
        count24_frame += 1
        elapsedtime = dt.datetime.today().timestamp() - start_time
        if elapsedtime >= 1:
            frames_per_second = count24_frame
            count24_frame = 0
            start_count = True

        # 6) Output: warp lane boundaries back & display lane boundaries, curvature and position
        lanes_marked, summary =  lanepipeline.draw_lanes(top_view, image, left_lane.current_fit, right_lane.current_fit, curvature,
                                      Minv, frame_position,  sanity_check, frames_per_second)
    
        # Set current values as previous values for next frame
        left_lane.previous_fit = left_lane.current_fit
        right_lane.previous_fit = right_lane.current_fit
    
        # Reset / empty current fit
        left_lane.current_fit = [np.array([False])]
        right_lane.current_fit = [np.array([False])]        
        
        canvas.addObj(CanvasObj('Final', 3, 4, lanes_marked ))

        canvas_img = canvas.render()
        cv2.imshow(nameApp, canvas_img)
        cv2.imshow(nameAppControlPanel, summary)
        canvas.clear()
        
    ####################
    #
    #
    # Keyboard control
    #
    #
    ####################
    key = cv2.waitKey(1)
    if key == 27: #ESC
        break   
    elif key & 0xFF == ord(' '): #or paused:
        paused = not paused
    elif key & 0xFF == ord('r'):
        if(paused): #only accepts rewind if is in pause
            rewind = True
            frame_position -= 1
            if frame_position < 1:
                frame_position = 1
    elif key & 0xFF == ord('f'):
        if(paused): #only accepts rewind if is in pause
            forward = True
            frame_position += 1
            if frame_position > total_frames:
                frame_position = total_frames

cap.release()
video_reader.release()
cv2.destroyAllWindows()
