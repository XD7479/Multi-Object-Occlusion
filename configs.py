import os, cv2, pickle
import numpy as np
import torch
import torch.cuda
import matplotlib.pyplot as plt

# field meanings are below

### ==== Setups ==== ###

device_ids = [2]

dataset_train = 'kins'     # pascal3d+, coco
dataset_eval = 'kins'      # occveh, coco, KINS

nn_type = 'resnext'             # vgg, resnext

vc_num = 512
K = 8
context_cluster = 5


### ==== Directories ==== ###

home_dir = os.getenv("HOME") + '/workspace/'
demo_dir = home_dir + 'demo/'
meta_dir = home_dir + 'meta/'
data_dir = home_dir + 'data/'
init_dir = meta_dir + 'init_{}/'.format(nn_type)
exp_dir = home_dir + 'exp/'

### ==== Categories ==== ###

categories = dict()
categories['pascal3d+'] = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
categories['occveh'] = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike']
# categories['occveh'] = ['boat', 'bottle', 'chair', 'diningtable', 'sofa', 'tvmonitor']
categories['coco'] = ["None", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", " parking meter", "bench", "bird", "cat", "dog",
                      "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                      "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                      "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                      "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                      "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                      "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                      "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                      "teddy bear", "hair drier", "toothbrush"]
categories['kins'] = ['_', 'cyclist', 'pedestrian', '_', 'car', 'tram', 'truck', 'van', 'misc']

categories['train'] = categories[dataset_train]
categories['eval']  = categories[dataset_eval]


### ==== Network ==== ###

if nn_type == 'vgg':
    layer = 'pool4'
    vMF_kappa = 30
    feature_num = 512
    feat_stride = 16

elif nn_type == 'resnext':
    layer = 'second'
    vMF_kappa = 65
    feature_num = 1024
    feat_stride = 16

else:
    print('Backbone Architecture Not Recognized')
    layer = ''
    vMF_kappa = 0
    feature_num = 0
    feat_stride = 0

rpn_configs = {'training_param' : {'weight_decay': 0.0005, 'lr_decay': 0.1, 'lr': 1e-3}, 'ratios' : [0.5, 1, 2], 'anchor_scales' : [8, 16, 32], 'feat_stride' : feat_stride }



'''

device_ids:         cuda device ids used
dataset_train:      dataset used for training
dataset_eval:       dataset used for evaluation
nn_type:            architecture type of backbone extractor
layer:              architecture layer of backbone extractor
feature_num:        number of channels of the architecture layer of backbone extractor
vMF_kappa:          the value used for the vMF activation
vc_num:             number of VC centers
context_cluster:    number of context centers
K:                  number of mixtures per category learned
categories:         map that contains various categories depending on the dataset
rpn_configs:        configs for the rpn training and evaluation
'''
