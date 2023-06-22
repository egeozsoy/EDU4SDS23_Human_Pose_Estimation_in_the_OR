#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://drive.google.com/uc?id=1Z3JvAFmL2IkBnQmmt5f4uTcXVhO5f7cq"/></center>
# 
# ------
# <center>&copy; Research Group CAMMA, University of Strasbourg, <a href="http://camma.u-strasbg.fr">http://camma.u-strasbg.fr</a>
# 
# <h2>Authors: Vinkle Srivastav, Idris Hamoud, Ege Oszoy </h2>
# </center>
# 
# ------

# # <center><font color=green> Lecture: Human Pose Estimation in the Operating Room </font></center>
# <center><img src="https://lh3.googleusercontent.com/drive-viewer/AFGJ81qU1PeDDhk9Tl0tfMwk9pMOGEz__WBAXMU6Fo4hZXgz20jwymqYgGL4PS6J0BxihQIeRmsv3nR14cgszU2YLKcYV6AefA=s1600"/></center>
# 
# 
# ### **Objectives**:
#   1. Understand how Pose estimation can be used to understand body language and non-verbal communication cues. It can be used in healthcare for patient monitoring or in this case in the Operating Room for workflow monitoring.
#   2. PyTorch `Dataset` and `Dataloader` for subset of MVOR dataset
#   3. Run off-the-shelf top-down approaches for HPE using Cascade Mask R-CNN as an object detector, and compare with groundtruth bounding boxes

# ## Setup

# In[ ]:


#@title Install external dependencies
# install dependencies
# !pip install numpy
# !pip install matplotlib
# !pip install torch
# !pip install torchvision
# !pip install tqdm
# !pip install ipywidgets
# !pip install -U openmim
# !mim install mmengine
# !mim install "mmcv>=2.0.0"
# !mim install "mmdet>=3.0.0"


# In[2]:


#@title Download resources
ROOT_DIR = "./HPE_dataset"
#!mkdir $ROOT_DIR && cd $ROOT_DIR && wget https://s3.unistra.fr/camma_public/datasets/mvor/camma_mvor_dataset.zip 
#!cd $ROOT_DIR && unzip -q camma_mvor_dataset.zip && rm camma_mvor_dataset.zip
#!cd $ROOT_DIR && wget https://raw.githubusercontent.com/CAMMA-public/MVOR/master/annotations/camma_mvor_2018.json
# Might need to upload a small subse of mvor only using 1 day out of the 4 daysof iimages


# In[118]:


#@title Imports
import os
#import sys
import json
import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch import optim
#import torchvision
import numpy as np
import torchvision.transforms as transforms
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi'] = 100
#from PIL import Image
import cv2
#import pickle
import random
import ipywidgets as wd
#import glob
#import io
import mmcv
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from mmcv import Config
from mmpose.apis import (inference_top_down_pose_model, inference_bottom_up_pose_model,
                         vis_pose_result, process_mmdet_results, init_pose_model)
from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO
#from mmpose.datasets import DatasetInfo

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True


# ## Data Exploration
# ### HPE helpers
# We will in a first step visualize the keypoint annotations on one sample image from the MVOR dataset. This example will allow us to better understand the different helper functions from the mmpose library and get a better understanding of the python visualization libraries like matplotlib.

# In[4]:


# Define the day and the name of the image we use for visualization
sample_frame_info = ("day1", "000013.png")
_,axs = plt.subplots(1,3,dpi=300)
# We will plot the multiview images from the three different cameras
for cam_id in range(3):
    rgb_img_path = os.path.join(ROOT_DIR, "camma_mvor_dataset", sample_frame_info[0],
                            "cam"+str(cam_id+1),"color", sample_frame_info[1])
    rgb_img = mmcv.imread(rgb_img_path, channel_order='rgb')
    axs[cam_id].imshow(rgb_img); axs[cam_id].set_title("Camera "+str(cam_id+1))


# In[183]:


# Helper function for visualization of depth maps
def normalize_depth_for_display(depth, pc=60, crop_percent=0., normalizer=None, cmap='viridis'):
    # convert to disparity
    depth = 1./(depth + 1e-6)
    if normalizer is not None:
        depth = depth/normalizer
    else:
        depth = depth/(np.percentile(depth, pc) + 1e-6)
    depth = np.clip(depth, 0, 1)
    depth = rgb2gray(depth)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]

    depth = depth
    return depth 


# In[185]:


# Define the day and the name of the image we use for visualization
sample_frame_info = ("day1", "000013.png")
_,axs = plt.subplots(1,3,dpi=300)
# We will plot the multiview images from the three different cameras
for cam_id in range(3):
    depth_img_path = os.path.join(ROOT_DIR, "camma_mvor_dataset", sample_frame_info[0],
                            "cam"+str(cam_id+1),"depth", sample_frame_info[1])
    depth_img = mmcv.imread(depth_img_path, channel_order='rgb')
    depth_img = normalize_depth_for_display(depth_img)
    axs[cam_id].imshow(depth_img); axs[cam_id].set_title("Camera "+str(cam_id+1))


# In[112]:


# Load annotations from json file
annot_path = os.path.join(ROOT_DIR, "camma_mvor_2018.json")


# ## Top-down approaches
# ### Human detection
# 

# 

# In[95]:


# Initialization of the human detector using mmdet api
det_config = '../mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_coco.py'
pose_config = '../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288.py'
det_model = init_detector(det_config, checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco/cascade_mask_rcnn_x101_64x4d_fpn_20e_coco_20200512_161033-bdb5126a.pth')
pose_model = init_pose_model(pose_config, checkpoint = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288-314c8528_20200708.pth')


# In[96]:


rgb_img_path =  os.path.join(ROOT_DIR, "camma_mvor_dataset", sample_frame_info[0],
                          "cam"+str(1),"color", sample_frame_info[1]) #'/raid/coco/images/test2017/000000000725.jpg'
rgb_img = mmcv.imread(rgb_img_path, channel_order='rgb')


# In[97]:


# test a single image, the resulting box is (x1, y1, x2, y2)
mmdet_results = inference_detector(det_model, rgb_img)
# keep the person class bounding boxes and process the results in an np.array format.
person_results = process_mmdet_results(mmdet_results, cat_id=1)
person_results = [{'bbox':np.array(box['bbox'].cpu()), 'device':'cpu'} for box in person_results]


# In[99]:


mmdet_results


# In[73]:


pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        rgb_img,
        person_results,
        bbox_thr=0.1,
        format='xyxy',
        return_heatmap=True)


# In[74]:


vis_result = vis_pose_result(
        pose_model,
        rgb_img,
        pose_results,
        out_file=os.path.join(ROOT_DIR,"test_pose.png"))

plt.imshow(vis_result)


# In[100]:


_,axs = plt.subplots(1,3,dpi=300)
for cam_id in range(3):
    rgb_img_path = os.path.join(ROOT_DIR, "camma_mvor_dataset", sample_frame_info[0],
                            "cam"+str(cam_id+1),"color", sample_frame_info[1])
    rgb_img = mmcv.imread(rgb_img_path, channel_order='rgb')
    # test a single image, the resulting box is (x1, y1, x2, y2)
    mmdet_results = inference_detector(det_model, rgb_img)
    # keep the person class bounding boxes and process the results in an np.array format.
    person_results = process_mmdet_results(mmdet_results, 1)
    person_results = [{'bbox':np.array(box['bbox'].cpu()), 'device':'cpu'} for box in person_results]
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        rgb_img_path,
        person_results,
        bbox_thr=0.,
        format='xyxy',
        return_heatmap=True)
    vis_result = vis_pose_result(
            pose_model,
            rgb_img,
            pose_results)
    axs[cam_id].imshow(vis_result); axs[cam_id].set_title("Camera "+str(cam_id+1))


# In[113]:


with open(annot_path) as f:
   data = json.load(f)


# In[119]:


# initialize COCO api for instance annotations
coco=COCO(annot_path)


# In[133]:


# Helper function to find ids for specific filename
def find_id(annots_gt, file_name):
    for img in annots_gt.imgs.values():
        if img['file_name']==file_name:
            return img['id']
        


# In[151]:


# Load annotations for sample image defined above
id_img = find_id(coco, sample_frame_info[0] + "/cam"+str(cam_id+1)+"/color/"+ sample_frame_info[1])
anns = coco.getAnnIds(id_img)
person_results = [{'bbox':ann['bbox']+[1.0]} for ann in coco.loadAnns(anns)]


# In[154]:


pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        rgb_img,
        person_results,
        bbox_thr=0.1,
        format='xywh',
        return_heatmap=True)


# In[155]:


vis_result = vis_pose_result(
        pose_model,
        rgb_img,
        pose_results,
        out_file=os.path.join(ROOT_DIR,"test_pose.png"))

plt.imshow(vis_result)


# ## Finetune model on MVOR ?

# In[ ]:


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
    cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
)
# valid_loader = torch.utils.data.DataLoader(
#     valid_dataset,
#     batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
#     shuffle=False,
#     num_workers=cfg.WORKERS,
#     pin_memory=True
# )

