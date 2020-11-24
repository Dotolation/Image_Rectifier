# importing the module 
from tkinter import *
from tkinter import filedialog as fd
from homography import rectify

import cv2
import numpy as np
   
# function to display the coordinates of 
# of the points clicked on the image
pre_warp = []
post_warp = []
count = 0
raw = None
original = None

def ResizeWithAspectRatio(image, height, width, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    ratio = 1

    if (h>height):
        ratio = height / h
    if (w > width):
        ratio = min(ratio, width / w)

    dim = (int(w*ratio), int(h*ratio))
    return (ratio, cv2.resize(image, dim, interpolation=inter))


def click_event(event, x, y, ignored, ratio):
    global pre_warp
    global count 
    global post_warp
    
    x_, y_ = (int(x / ratio), int(y / ratio))

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0,0,0)
    message = ""
    
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN:
        count += 1

        if(count <= 4):
            pre_warp.append([x_,y_])
            color = (255,0,0)
            message = str(count)
        else:
            post_warp.append([x_,y_])
            color = (168,168, 0)
            message = str(count-4)
            
        if (count < 8):
            #print(x_, ' ', y_)
            cv2.putText(resized, message, (x,y), font, 
                    1, color , 2) 
            cv2.imshow("image", resized)
            
        else:
            cv2.destroyAllWindows()
            
            print("Processing the Homography. Please wait.")
            planar_c, target_c = (np.array(pre_warp), np.array(post_warp))
            rect = rectify(raw, planar_c.astype(float), target_c.astype(float)).astype(np.uint8)
            after = ResizeWithAspectRatio(rect, 720, 1280)[1]
            cv2.imshow("Before", original)
            cv2.imshow("After", after)
            print("done!")
            cv2.imwrite('rectified.jpg', rect)


  
def get_coords():
    '''
    #1. File Selection.
    
    '''
    global pre_warp
    global post_warp
    global resized
    global original
    global raw
    
    root = Tk()
    root.filename = fd.askopenfilename(title="Select the image to rectify",\
                                       filetypes=(("Images", ".png .jpeg .jpg .bmp"),\
                                                  ("GIF files", ".gif")))
    '''
    #2. Point Selection / Warping 
    '''
    
    #Reading & Resizing The Image 
    raw = cv2.imread(root.filename, cv2.IMREAD_COLOR)
    ratio, resized = ResizeWithAspectRatio(raw, 720, 1280)
    print("ratio: ", ratio)

    #Clickable Display
    original = resized.copy()
    cv2.imshow('image', resized)
    cv2.setMouseCallback('image', click_event, ratio)
    

if __name__ == "__main__":
    get_coords()




    
    

    
    
    
  

