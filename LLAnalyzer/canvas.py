# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 23:22:30 2020

@author: Patrick de Faria
"""

import numpy as np
import cv2

class CanvasObj:
    def __init__(self, title, row, col, img):
        self.img = img
        self.col = col
        self.row = row
        self.title = title

class Canvas:
    def __init__(self, canvasheight= 864, canvaswidth=1740, objheight = 216, objwidth = 384, 
                 initial_leftpos = 25, initial_toppos = 25, objmarginx = 50, objmarginy = 50):
        self.canvasheight = canvasheight
        self.canvaswidth = canvaswidth
        self.objheight = objheight 
        self.objwidth = objwidth
        self.initial_leftpos = initial_leftpos
        self.initial_toppos = initial_toppos
        self.objmarginx = objmarginx
        self.objmarginy = objmarginy

        self.list = [] 

        self.canvas_img = np.zeros([canvasheight, canvaswidth,3],dtype=np.uint8)
        self.canvas_img.fill(60)
        
    def addObj(self, canvasobj):            
        self.list.append(canvasobj)
        return

    def clear(self):
        del self.list[:]

    # render the objects
    def render(self):
        for obj in self.list:
            y1 = self.initial_toppos + ((obj.row - 1) * self.objheight) + ( (obj.row - 1) * self.objmarginy)
            y2 = y1 + self.objheight
            
            x1 = self.initial_leftpos + ((obj.col - 1) * self.objwidth) + ( (obj.col - 1) * self.objmarginx)
            x2 = x1 + self.objwidth

            self.canvas_img[y1:y2, x1:x2, :] = obj.img
            
            cv2.putText(self.canvas_img, obj.title, (x1, y1-6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 144, 30), 1)
            
        return self.canvas_img
