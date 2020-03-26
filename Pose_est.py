
from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

ctx = mx.cpu()
detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)


detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
detector.hybridize()

estimator = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)
estimator.hybridize()

#cap = cv2.VideoCapture(0)
frame=cv2.imread("temp/w1.jpg")
time.sleep(1)  ### letting the camera autofocus


axes = None
num_frames = 1000

for i in range(num_frames):
    #ret, frame = cap.read()
    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')

    x, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(frame, class_IDs, scores, bounding_boxs,
                                                        output_shape=(128, 96), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = estimator(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        img = cv_plot_keypoints(frame, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=0.5, keypoint_thresh=0.2)

    
    cv_plot_image(img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

