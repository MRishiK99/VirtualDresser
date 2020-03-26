

from __future__ import division

import time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform
from gluoncv.utils.metrics import HeatmapAccuracy


# Loading the data

train_dataset = mscoco.keypoints.COCOKeyPoints('~/.mxnet/datasets/coco',
                                               splits=('person_keypoints_train2017'))



# For augmentation, we randomly scale, rotate or flip the input.
# Finally we normalize it with the standard ImageNet statistics.

# The COCO keypoints dataset contains 17 keypoints for a person.
# Each keypoint is annotated with three numbers ``(x, y, v)``, where ``x`` and ``y``
# mark the coordinates, and ``v`` indicates if the keypoint is visible.

# For each keypoint, we generate a gaussian kernel centered at the ``(x, y)`` coordinate, and use
# it as the training label. This means the model predicts a gaussian distribution on a feature map.


transform_train = SimplePoseDefaultTrainTransform(num_joints=train_dataset.num_joints,
                                                  joint_pairs=train_dataset.joint_pairs,
                                                  image_size=(256, 192), heatmap_size=(64, 48),
                                                  scale_factor=0.30, rotation_factor=40, random_flip=True)


batch_size = 32
train_data = gluon.data.DataLoader(
    train_dataset.transform(transform_train),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=8)



# Deconvolution Layer




# Model Definition

# We load the pre-trained parameters for the ``ResNet18`` layers,
# and initialize the deconvolution layer and the final convolution layer.

context = mx.gpu(0)
net = get_model('simple_pose_resnet18_v1b', num_joints=17, pretrained_base=True,
                ctx=context, pretrained_ctx=context)
net.deconv_layers.initialize(ctx=context)
net.final_layer.initialize(ctx=context)


# summary of the model

x = mx.nd.ones((1, 3, 256, 192), ctx=context)
net.summary(x)



L = gluon.loss.L2Loss()

# - Learning Rate Schedule and Optimizer:


num_training_samples = len(train_dataset)
num_batches = num_training_samples // batch_size
lr_scheduler = LRScheduler(mode='step', base_lr=0.001,
                           iters_per_epoch=num_batches, nepochs=140,
                           step_epoch=(90, 120), step_factor=0.1)


#     ``adam`` as the optimizer.


trainer = gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler': lr_scheduler})

metric = HeatmapAccuracy()



# Training Loop


net.hybridize(static_alloc=True, static_shape=True)
for epoch in range(3):
    metric.reset()

    for i, batch in enumerate(train_data):
        if i > 0:
            break
        data = gluon.utils.split_and_load(batch[0], ctx_list=[context], batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=[context], batch_axis=0)
        weight = gluon.utils.split_and_load(batch[2], ctx_list=[context], batch_axis=0)

        with ag.record():
            outputs = [net(X) for X in data]
            loss = [L(yhat, y, w) for yhat, y, w in zip(outputs, label, weight)]

        for l in loss:
            l.backward()
        trainer.step(batch_size)

        metric.update(label, outputs)

    break
