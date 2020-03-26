
from __future__ import division
import argparse
import time
import logging
import os
import sys
import math
import tqdm
import cv2
import imutils
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt
import math  

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

shirt_enabled=1
pant_enabled=1

def Distance(x1,y1,x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist  

def shrink(img):
    for top in range(img.shape[0]):
        if img[top,:].any():
            break

    for bottom in range(top,img.shape[0]):
        if not img[bottom,:].any():
            break
    for left in range(img.shape[1]):
        if img[:,left].any():
            break
    for right in range(left,img.shape[1]):
        if not img[:,right].any():
            break
    shrinked = img[top:bottom,left:right]
    return shrinked

def validatex(x,cfd,i):
    if type(x) == type(None) or x>262 or x<0 or cfd<0.8:
        if i >= 11 and i<=16:
            pant_enabled=0
        else:
            shirt_enabled=0
        return 0
    else:
        return x

def validatey(x,cfd,i):
    if type(x) == type(None) or x>350 or x<0 or cfd<0.8:
        if i >= 11 and i<=16:
            pant_enabled=0
        else:
            shirt_enabled=0
        return 0
    else:
        return x

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the ran
    ge [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

def changecolor(img,color):
    for i in range(0,img.shape[0]-1):
        for j in range(0,img.shape[1]-1):
            if img[i][j].any():
                if img[i][j][0]+color[0]<=255 and img[i][j][0]+color[0]>=0:
                    img[i][j][0] = (img[i][j][0]+color[0])   #B 72
                if img[i][j][1]+color[1]<=255 and img[i][j][1]+color[1]>=0:
                    img[i][j][1] = (img[i][j][1]+color[1])     #G 83
                if img[i][j][2]+color[2]<=255 and img[i][j][2]+color[2]>=0:
                    img[i][j][2] = (img[i][j][2]+color[2])
                img[i][j][3] = img[i][j][3]
    return img

##########  COLOR RULES ##############
#### SHIRTCOLOR
#### (100,120,0) - CREME
#### (0,120,0) - YELLOW
#### (80,50,-100) - BLUE
#### (100,0,0) - PINK
shirt_color_dict = {'0':[0,0,0],'1':[100,120,0],'2':[0,120,0],'3':[80,50,-100],'4':[100,0,0]}
shirt_size_dict = {'1':0.9,'2':1,'3':1.1,'4':1.3,'5':1.4}
pant_size_dict = {'1':0.9,'2':1,'3':1.1,'4':1.3,'5':1.4}
#### PANTCOLOR
#### (100,100,100) - WHITE
#### (100,0,0) - BLUE
#### (0,100,0) - GREEN
#### (0,0,100) - RED
#### (-40,-20,20) - BROWN
pant_color_dict = {'0':[0,0,0],'1':[100,100,100],'2':[100,0,0],'3':[0,100,0],'4':[0,0,100],'5':[-40,-20,20]}

ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person': 'person'})
detector.hybridize()

estimator = get_model('simple_pose_resnet18_v1b',
                      pretrained='ccd24037', ctx=ctx)
estimator.hybridize()

shirt_size=1
pant_size=1
shirt_color=[0,0,0]
pant_color=[0,0,0]

args = sys.argv[1:]
shirt_enabled = 1 if args[0]!='-1' else 0
pant_enabled = 1 if args[1]!='-1' else 0
glass_enabled = 1 if args[2]!='-1' else 0

if shirt_enabled:
    shirt_color = shirt_color_dict[args[0]]
    shirt_size = shirt_size_dict[args[3]]

if pant_enabled:
    pant_color = pant_color_dict[args[1]]
    pant_size = pant_size_dict[args[4]]

if glass_enabled:
    glass_ch = args[2]

cap = cv2.VideoCapture(0)
time.sleep(1)  # letting the camera autofocus

axes = None
num_frames = 1000

for i in range(num_frames):
    ret,frame = cap.read()
    #frame = cv2.imread("w1.jpg")

    if type(frame) == type(None):
        break
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    x, frame = gcv.data.transforms.presets.ssd.transform_test(
        frame, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)
    size=1.2
    glass=3
    shirtcolor = [100,140,-200]
    #shirtcolor = [reqcolor[i]-originalcolor[i] for i in range(0,3)]
    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                    output_shape=(128, 96), ctx=ctx)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(
            predicted_heatmap, upscale_bbox)

        
        
        ###########################################################
        ###### RIGHT - RIGHT END OF THE IMAGE
        ###### LEFT  - LEFT END OF THE IMAGE
        ### 0  - NOSE
        ### 1  - RIGHT EYE
        ### 2  - LEFT EYE
        ### 3  - RIGHT EAR
        ### 4  - LEFT EAR
        ### 5  - RIGHT SHOULDER
        ### 6  - LEFT SHOULDER
        ### 7  - RIGHT ELBOW
        ### 8  - LEFT ELBOW
        ### 9  - RIGHT WRIST
        ### 10 - LEFT WRIST
        ### 11 - RIGHT HIP
        ### 12 - LEFT HIP
        ### 13 - RIGHT KNEE
        ### 14 - LEFT KNEE
        ### 15 - RIGHT FOOT
        ### 16 - LEFT FOOT
        ############################################################
        for i in range(0,17):
            pred_coords[0][i].asnumpy()[0]=validatex(pred_coords[0][i].asnumpy()[0],confidence[0][i].asnumpy(),i)
            pred_coords[0][i].asnumpy()[1]=validatey(pred_coords[0][i].asnumpy()[1],confidence[0][i].asnumpy(),i)

        noseX=pred_coords[0][0].asnumpy()[0]
        noseY=pred_coords[0][0].asnumpy()[1]
        
        right_eyeX=pred_coords[0][1].asnumpy()[0]
        right_eyeY=pred_coords[0][1].asnumpy()[1]

        left_eyeX=pred_coords[0][2].asnumpy()[0]
        left_eyeY=pred_coords[0][2].asnumpy()[1]

        right_earX=pred_coords[0][3].asnumpy()[0]
        right_earY=pred_coords[0][3].asnumpy()[1]

        left_earX=pred_coords[0][4].asnumpy()[0]
        left_earY=pred_coords[0][4].asnumpy()[1]

        rshX=pred_coords[0][5].asnumpy()[0]
        rshY=pred_coords[0][5].asnumpy()[1]

        lshX=pred_coords[0][6].asnumpy()[0]
        lshY=pred_coords[0][6].asnumpy()[1]

        relbX=pred_coords[0][7].asnumpy()[0]
        relbY=pred_coords[0][7].asnumpy()[1]

        lelbX=pred_coords[0][8].asnumpy()[0]
        lelbY=pred_coords[0][8].asnumpy()[1]

        rwristX=pred_coords[0][9].asnumpy()[0]
        rwristY=pred_coords[0][9].asnumpy()[1]

        lwristX=pred_coords[0][10].asnumpy()[0]
        lwristY=pred_coords[0][10].asnumpy()[1]

        rhipX=pred_coords[0][11].asnumpy()[0]
        rhipY=pred_coords[0][11].asnumpy()[1]

        lhipX=pred_coords[0][12].asnumpy()[0]
        lhipY=pred_coords[0][12].asnumpy()[1]

        rkneeX=pred_coords[0][13].asnumpy()[0]
        rkneeY=pred_coords[0][13].asnumpy()[1]

        lkneeX=pred_coords[0][14].asnumpy()[0]
        lkneeY=pred_coords[0][14].asnumpy()[1]

        rfootX=pred_coords[0][15].asnumpy()[0]
        rfootY=pred_coords[0][15].asnumpy()[1]

        lfootX=pred_coords[0][16].asnumpy()[0]
        lfootY=pred_coords[0][16].asnumpy()[1]

        # for i in range(0,17):
        #    cv2.putText(frame, str(i), (pred_coords[0][i].asnumpy()[0], pred_coords[0][i].asnumpy()[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,box_thresh=0.5, keypoint_thresh=0.2)
        
        ######################### MID #########################
        
        if rshX>0 and rhipX>0 and lshX>0 and lhipX>0: 
            shwidth=Distance(rshX,rshY,lshX,lshY)
            umidX = (rshX+lshX)/2
            umidY = (rshY+lshY)/2
            lmidX = (rhipX+lhipX)/2
            lmidY = (rhipY+lhipY)/2
            angle=(math.atan2(umidY-lmidY, umidX-lmidX) * 180 / 3.14)-270
            body_height=Distance(umidX,umidY,lmidX,lmidY)
            s_img = cv2.imread("images/shirt.png", -1)
            h=s_img.shape[0]
            w=s_img.shape[1]
            width = int(shwidth*shirt_size)
            height = int(body_height*shirt_size)
            verticalr = height/h
            horizontalr = width/w
            if height>0 and width>0:
                s_img=cv2.resize(s_img,(width,height))
                #s_img = cv2.flip(s_img,1)
                s_img = imutils.rotate_bound(s_img, angle)
                s_img = shrink(s_img)
                midH = s_img.shape[0]

                x=int(lshX-50*horizontalr)
                y=int(lshY-160*verticalr)
                midX=x
                midY=y

                cloth_mid = changecolor(s_img,shirt_color)       
        
        ######################### LSH ###########################
        x1=lelbX   #LEFT ELBOW
        y1=lelbY   #LEFT ELBOW
        x2=lshX   #LEFT SHOULDER
        y2=lshY   #LEFT SHOULDER
        
        if lshX>0 and lelbX>0:
            angle=(math.atan2(y2-y1, x2-x1) * 180 / 3.14)-270-20
            lshwidth=Distance(x1,y1,x2,y2)
            s_img = cv2.imread("images/lsh.png", -1)
            #height = int(lshwidth*shirt_size)
            #width = int(s_img.shape[1]*(height/s_img.shape[0]*shirt_size))
            height = int(s_img.shape[0]*verticalr*1.5)
            width = int(s_img.shape[1]*horizontalr)

            if height>0 and width>0:
                s_img=cv2.resize(s_img,(width,height))
                s_img = imutils.rotate_bound(s_img, angle)

                s_img = shrink(s_img)
                
                cloth_lshx=int(pred_coords[0][6].asnumpy()[0]-s_img.shape[1]/1.5*shirt_size)
                cloth_lshy=int(pred_coords[0][6].asnumpy()[1]-height/10*shirt_size)

                cloth_lsh = changecolor(s_img,shirt_color)
                
                

        ########################## RSH ####################
        x1=rshX   #RIGHT SHOULDER
        y1=rshY   #RIGHT SHOULDER
        x2=relbX   #RIGHT ELBOW
        y2=relbY   #RIGHT ELBOW

        if rshX>0 and relbX>0:
            angle=(math.atan2(y1-y2, x1-x2) * 180 / 3.14)-270+20
            rshwidth=Distance(x1,y1,x2,y2)
            s_img = cv2.imread("images/rsh.png", -1)
            #height = int(rshwidth*shirt_size)
            #width = int(s_img.shape[1]*(height/s_img.shape[0])*shirt_size)
            height = int(s_img.shape[0]*verticalr*1.5)
            width = int(s_img.shape[1]*horizontalr)
            if height>0 and width>0:
                s_img=cv2.resize(s_img,(width,height))
                #s_img = cv2.flip(s_img,1)
                s_img = imutils.rotate_bound(s_img, angle)
                s_img = shrink(s_img)
                cloth_rshx=int(pred_coords[0][5].asnumpy()[0]-width/(8*shirt_size))
                cloth_rshy=int(pred_coords[0][5].asnumpy()[1]-height/(6*shirt_size))

                cloth_rsh = changecolor(s_img,shirt_color)

        if pant_enabled:
            ######################### PANTS - UPPER left THIGH #################
            if rshX>0 and rhipX>0 and lshX>0 and lhipX>0 and lkneeX>0 and rkneeX>0:
                hipwidth=Distance(rshX,rshY,lshX,lshY)
                angle=(math.atan2(lhipY-lkneeY, lhipX-lkneeX) * 180 / 3.14) - 270
                th_height=(Distance(lhipX,lhipY,lkneeX,lkneeY))*1.2
                s_img = cv2.imread("images/leftthigh.png", -1)
                h=s_img.shape[0]
                w=s_img.shape[1]
                width = int(hipwidth*pant_size*1.2/2)
                height = int(th_height*pant_size)
                lthh = height
                lthw = width
                verticalr = height/h
                horizontalr = width/w
                if height>0 and width>0:
                    s_img=cv2.resize(s_img,(width,height))
                    #s_img = cv2.flip(s_img,1)
                    s_img = imutils.rotate_bound(s_img, angle)
                    s_img = changecolor(s_img,pant_color)

                    cloth_ulth = shrink(s_img)

                    x=int(lshX-20*horizontalr)
                    y=int(lhipY)
                    lthx=x
                    lthy=y

            ######################### PANTS - UPPER right THIGH #################
            if rshX>0 and rhipX>0 and lshX>0 and lhipX>0 and lkneeX>0 and rkneeX>0:
                hipwidth=Distance(rshX,rshY,lshX,lshY)
                angle=(math.atan2(rhipY-rkneeY, rhipX-rkneeX) * 180 / 3.14) - 270
                th_height=(Distance(rhipX,rhipY,rkneeX,rkneeY))*1.2
                s_img = cv2.imread("images/rightthigh.png", -1)
                h=s_img.shape[0]
                w=s_img.shape[1]
                #width = int(hipwidth*pant_size*1.2/2)
                #height = int(th_height*pant_size)
                height = int(h*verticalr)
                width = int(w*horizontalr)
                rthh = height
                rthw = width
                verticalr = height/h
                horizontalr = width/w
                if height>0 and width>0:
                    s_img=cv2.resize(s_img,(width,height))
                    #s_img = cv2.flip(s_img,1)
                    s_img = imutils.rotate_bound(s_img, angle)
                    s_img = changecolor(s_img,pant_color)
                    cloth_urth = shrink(s_img)

                    x=int(lthw+ lthx)
                    y=int(lhipY)
                    rthx=x
                    rthy=y
                
            ########################## LEFT KNEE ###################
            if lkneeX>0 and lfootX>0:
                s_img = cv2.imread("images/leftknee.png", -1)
                lknee_height=Distance(lkneeX,lkneeY,lfootX,lfootY)
                h=s_img.shape[0]
                w=s_img.shape[1]
                #height = int(lknee_height*pant_size/1.25)
                #width = int((lthw/1.5)*pant_size)
                height = int(h*verticalr)
                width = int(w*horizontalr)
                angle=(math.atan2(lkneeY-lfootY, lkneeX-lfootX) * 180 / 3.14)-270

                if height>0 and width>0:
                    s_img=cv2.resize(s_img,(width,height))
                    #s_img = cv2.flip(s_img,1)
                    s_img = changecolor(s_img,pant_color)

                    cloth_lknee = imutils.rotate_bound(s_img, angle)
                    

                    lknx=int(lkneeX-width/6)
                    lkny=int(lkneeY+height/20)

            ########################## RIGHT KNEE ###################
            if rkneeX>0 and rfootX>0 :
                s_img = cv2.imread("images/rightknee.png", -1)
                rknee_height=Distance(rkneeX,rkneeY,rfootX,rfootY)
                h=s_img.shape[0]
                w=s_img.shape[1]
                #height = int(rknee_height*pant_size/1.25)
                #width = int((rthw/1.5)*pant_size)
                height = int(h*verticalr)
                width = int(w*horizontalr)
                angle=(math.atan2(rkneeY-rfootY, rkneeX-rfootX) * 180 / 3.14)-270
                if height>0 and width>0:
                    s_img=cv2.resize(s_img,(width,height))
                    #s_img = cv2.flip(s_img,1)
                    s_img = imutils.rotate_bound(s_img, angle)
                    s_img = changecolor(s_img,pant_color)

                    cloth_rknee = shrink(s_img)

                    rknx=int(rkneeX-width/6)
                    rkny=int(rkneeY+height/20)
            
        ############################ COOLERS ###################
        if glass_enabled:
            if right_eyeX>0 and left_eyeX>0 and right_earX>0 and left_earX>0:
                s_img = cv2.imread("images/glasses"+glass_ch+".png",-1)
                width = int(Distance((right_earX+right_eyeX)/2,(right_earY+right_eyeY)/2,(left_earX+left_eyeX)/2,(left_earY+left_eyeY)/2)*size)
                height = int(s_img.shape[0]*width/s_img.shape[1]*size)
                angle = (math.atan2(left_eyeY-right_eyeY,left_eyeX-right_eyeX)*180/3.14)-180
                verticalr = height/s_img.shape[0]

                if height>0 and width>0:
                    s_img=cv2.resize(s_img,(width,height))
                    #s_img = cv2.flip(s_img,1)
                    s_img = imutils.rotate_bound(s_img, angle)
                    cloth_glass = shrink(s_img)
                    if glass == 3:
                        glassx = int((left_earX+left_eyeX)/2-20*verticalr)
                        glassy = int((left_earY+left_eyeY)/2-30*verticalr) 
                    else:   
                        glassx = int((left_earX+left_eyeX)/2-50*verticalr)
                        glassy = int((left_earY+left_eyeY)/2-150*verticalr)
        ############################################################
        
        if glass_enabled and type(cloth_glass) != type(None) :
            overlay_image_alpha(frame,
                            cloth_glass[:, :, 0:3],
                            (int(glassx), int(glassy)),
                            cloth_glass[:, :, 3] / 255.0)
        if shirt_enabled and type(cloth_mid) != type(None) :
            overlay_image_alpha(frame,
                            cloth_mid[:, :, 0:3],
                            (int(midX), int(midY)),
                            cloth_mid[:, :, 3] / 255.0)
        if shirt_enabled and type(cloth_lsh) != type(None) :
            overlay_image_alpha(frame,
                            cloth_lsh[:, :, 0:3],
                            (int(cloth_lshx), int(cloth_lshy)),
                            cloth_lsh[:, :, 3] / 255.0)
        if shirt_enabled and type(cloth_rsh) != type(None) :
            overlay_image_alpha(frame,
                            cloth_rsh[:, :, 0:3],
                            (int(cloth_rshx), int(cloth_rshy)),
                            cloth_rsh[:, :, 3] / 255.0)    
        if pant_enabled and type(cloth_lknee) != type(None) :
            overlay_image_alpha(frame,
                            cloth_lknee[:, :, 0:3],
                            (int(lknx), int(lkny)),
                            cloth_lknee[:, :, 3] / 255.0)
        if pant_enabled and type(cloth_rknee) != type(None) :
            overlay_image_alpha(frame,
                            cloth_rknee[:, :, 0:3],
                            (int(rknx), int(rkny)),
                            cloth_rknee[:, :, 3] / 255.0)
        
        if pant_enabled and type(cloth_ulth) != type(None) :
            overlay_image_alpha(frame,
                            cloth_ulth[:, :, 0:3],
                            (int(lthx), int(lthy)),
                            cloth_ulth[:, :, 3] / 255.0)
        if pant_enabled and type(cloth_urth) != type(None) :
            overlay_image_alpha(frame,
                            cloth_urth[:, :, 0:3],
                            (int(rthx), int(rthy)),
                            cloth_urth[:, :, 3] / 255.0)

        
    cv2.imshow("img",frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
