import numpy as np
import math
import argparse
import imutils
import cv2


def correct_black(image):
    #make black pixels purple
    mask_a = image.copy()
    lower = np.array([0,0,0]) 
    upper = np.array([0,0,0]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(55,55,55)

    # make white pixels black
    lower = np.array([254,254,254]) 
    upper = np.array([256,256,256]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(0,0,0)

    #make red pixels black
    lower = np.array([5,5,225]) 
    upper = np.array([5,5,225]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(0,0,0)

    #make purple pixels white
    lower = np.array([55,55,55]) 
    upper = np.array([55,55,55]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(255,255,255)


    #find white contours

    gray = cv2.cvtColor(mask_a, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    num=0
   
    masksegment=image.copy()
    mask_final=image.copy()
    cunts = []
    for c in cnts:
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0]) 
        M = cv2.moments(c)
        if (M["m00"]>0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = leftmost[0]
            cY= leftmost[1]
       
        cnt_norm = c - [cX, cY]
        cnt_scaled = cnt_norm * 1.1
        cnt_scaled = cnt_scaled + [cX, cY]
        cnt_scaled = cnt_scaled.astype(np.int32)
        cunts.append(cnt_scaled)

    for c in cunts:
            
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0]) 
        horizontal = rightmost[0]- leftmost[0] 
        vertikal = bottommost[1]-topmost[1]
                
        if (topmost[1]<=(bottommost[1]) and leftmost[0]<=(rightmost[0])):
           
            M = cv2.moments(c)
            if (M["m00"]>0):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = leftmost[0]
                cY= leftmost[1]
            
            r,g,b = mask_final[cY, cX]
            
            if(r==0 and g==0 and b==0 and horizontal<400):
            
                cimg= np.zeros_like(masksegment)
                cv2.drawContours(cimg, cunts, num, color=24, thickness=8)
                cv2.drawContours(cimg, cunts, num, color=24, thickness=-1)
                pts = np.where(cimg == 24)
                masksegment[pts[0],pts[1]] = [55,55,55]
                add = 5
                crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]
           
                if not (np.any(crop_mask==[0,0,0])): 
                    #faulty black superpixel surrounded by white is then colored white
                    masksegment[pts[0],pts[1]] = [255,255,255]
                    crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]
                    if ( np.all(crop_mask==[255,255,255])):        
                        print("Faulty black")
                        image[pts[0],pts[1]] = [255,255,255] 
                    
                    #faulty black surrounded by red is then colored red
                    masksegment[pts[0],pts[1]] = [5,5,225]
                    crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]
                    if ( np.all(crop_mask==[5,5,225])):        
                        print("Faulty black")
                        image[pts[0],pts[1]] = [5,5,225]
                    
        num=num+1
    return image


def correct_white(image):
    
    #make white pixel purple
    mask_a = image.copy()
    lower = np.array([255,255,255]) 
    upper = np.array([255,255,255]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(55,55,55)

    #make red pixel black
    lower = np.array([5,5,225]) 
    upper = np.array([5,5,225]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(0,0,0)

    #make purple pixel white 
    lower = np.array([55,55,55]) 
    upper = np.array([55,55,55]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(255,255,255)


    #find white contours

    gray = cv2.cvtColor(mask_a, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    num=0
    masksegment=image.copy()
    mask_final=image.copy()
    cunts = []
    for c in cnts:
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0]) 
        M = cv2.moments(c)
        if (M["m00"]>0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = leftmost[0]
            cY= leftmost[1]

        cnt_norm = c - [cX, cY]
        cnt_scaled = cnt_norm * 1.1
        cnt_scaled = cnt_scaled + [cX, cY]
        cnt_scaled = cnt_scaled.astype(np.int32)
        cunts.append(cnt_scaled)

    for c in cunts:
        
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0]) 
        horizontal = rightmost[0]- leftmost[0] 
        vertikal = bottommost[1]-topmost[1]
 
        if (topmost[1]<=(bottommost[1]) and leftmost[0]<=(rightmost[0])):
           
            
            M = cv2.moments(c)
            if (M["m00"]>0):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = leftmost[0]
                cY= leftmost[1]
            
            r,g,b = mask_final[cY, cX]
            
            
            if(r==255 and g==255 and b==255 and horizontal<400):
            
                cimg= np.zeros_like(masksegment)
                cv2.drawContours(cimg, cunts, num, color=24, thickness=8)
                cv2.drawContours(cimg, cunts, num, color=24, thickness=-1)
                pts = np.where(cimg == 24)
                masksegment[pts[0],pts[1]] = [55,55,55]
                add = 20
                crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]

                if not ( np.any(crop_mask==[255,255,255])):  
                    #make white superpixel surrounded by black also black
                    masksegment[pts[0],pts[1]] = [0,0,0]
                    crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]
                    if ( np.all(crop_mask==[0,0,0])):        
                        print("Faulty white")
                        image[pts[0],pts[1]] = [0,0,0]
                    #make white superpixel surrounded by red also red
                    masksegment[pts[0],pts[1]] = [5,5,225]
                    crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]
                    if ( np.all(crop_mask==[5,5,225])):        
                        print("Faulty white")
                        image[pts[0],pts[1]] = [5,5,225]
                    
                r,g,b = masksegment[cY, cX]
  
        num=num+1
    return image


def correct_red(image):

    #make red pixel purple
    mask_a = image.copy()
    lower = np.array([5,5,225]) 
    upper = np.array([5,5,225]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(55,55,55)

    #make white pixel black
    lower = np.array([255,255,255]) 
    upper = np.array([255,255,225]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(0,0,0)

    #make purple pixel white
    lower = np.array([55,55,55]) 
    upper = np.array([55,55,55]) 
    single=0
    single = cv2.inRange(mask_a, lower, upper )
    mask_a[single>0]=(255,255,255)


    #find white contours
    gray = cv2.cvtColor(mask_a, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    num=0
    masksegment=image.copy()
    mask_final=image.copy()
    cunts = []
    for c in cnts:
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0]) 
        M = cv2.moments(c)
        if (M["m00"]>0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = leftmost[0]
            cY= leftmost[1]
        
        ## blowing up the contour a little
        cnt_norm = c - [cX, cY]
        cnt_scaled = cnt_norm * 1.1
        cnt_scaled = cnt_scaled + [cX, cY]
        cnt_scaled = cnt_scaled.astype(np.int32)
        cunts.append(cnt_scaled)

    for c in cunts:
                
        leftmost = tuple(c[c[:,:,0].argmin()][0])
        rightmost = tuple(c[c[:,:,0].argmax()][0])
        topmost = tuple(c[c[:,:,1].argmin()][0])
        bottommost = tuple(c[c[:,:,1].argmax()][0]) 
        horizontal = rightmost[0]- leftmost[0] 
        vertikal = bottommost[1]-topmost[1]
                        
        if (topmost[1]<=(bottommost[1]) and leftmost[0]<=(rightmost[0])):
            M = cv2.moments(c)
            if (M["m00"]>0):
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = leftmost[0]
                cY= leftmost[1]            
            r,g,b = image[cY, cX]
            
            if(r==5 and g==5 and b==225 and horizontal<400):
            
                cimg= np.zeros_like(image.copy())
                cv2.drawContours(cimg, cunts, num, color=24, thickness=8)
                cv2.drawContours(cimg, cunts, num, color=24, thickness=-1)
                pts = np.where(cimg == 24)
                masksegment[pts[0],pts[1]] = [55,55,55]
                add = 5
                crop_mask = masksegment[(topmost[1]-add):(bottommost[1]+add),(leftmost[0]-add):(rightmost[0]+add)]
                # wmake red superpixel black if its not surrounded by white or red superpixels         
                if not (np.any(crop_mask==[5,5,225]) or np.any(crop_mask==[255,255,255])): 
                    print("Faulty red")
                    mask_final[pts[0],pts[1]] = [0,0,0]
              
        num=num+1
    return mask_final

if __name__ == "__main__":

    mask = cv2.imread('/home/janischl/ssn-pytorch/results/8_7_4_result_retrained.png') 
   
    mask = correct_red(mask)    ## corrects faulty red (tool) superpixels that are completely spourrounded by black superpixels
    mask = correct_black(mask)  ## corrects faulty black (background) superpixels that are completely spourrounded by red or white superpixels
    mask = correct_white(mask)  ## corrects faulty white (wear) superpixels that are completely spourrounded by black or red superpixels (red superpixels sourrounding is actually possible to be right)
    cv2.imwrite(('final_correct.png'),mask)
