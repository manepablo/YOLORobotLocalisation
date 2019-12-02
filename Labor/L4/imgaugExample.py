#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:02:41 2019

@author: root
"""

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from matplotlib import pyplot as plt
import numpy as np

def visualizeImageAndMask(image,mask):
    plt.figure()
    plt.subplot(1,2,1)
    plt.title("image")
    plt.imshow(image)
    
    plt.subplot(1,2,2)
    plt.title("masks")
    plt.imshow(np.array(mask,dtype=np.float32))
    
# load some images and corresponding masks from the training data (different size)
x = np.array([cv2.imread("C:/Users/Paul/Desktop/DLM/images/" + str(i) + ".png")for i in range(0,8)])
y = np.array([cv2.imread("C:/Users/Paul/Desktop/DLM/masks/" + str(i) + ".png")for i in range(0,8)])

visualizeImageAndMask(x[0],y[0])

# example for having different sizes across the images.

# define the augmentation
seq1 = iaa.Sequential([ iaa.Affine(rotate=(-30,30))]) # Augmentation for images and masks
seq2 = iaa.Sequential([ iaa.Dropout([0.1,0.5])])      # Augmentation for images

"""
    Method 1
"""

# It feels clumsy cause we have to define a pipeline and calling different augmenters
# However, we can access the image directly
for img, mask in zip(x,y):
    seq1.deterministic = True
    
    image_augmented = seq1.augment_image(image=img)
    final_image = seq2.augment_image(image=image_augmented)
    
    mask_augmented = seq1.augment_image(image=mask)
    
    visualizeImageAndMask(final_image,mask_augmented)

"""
    Method 2
"""

# Here we can describe the Augmenter as one, however we need to access the image
# over ".get_arr()"
seq = iaa.Sequential([iaa.Affine(rotate=(-30,30)),iaa.Dropout([0.1,0.5])])

for img,mask in zip(x,y):
    segmap = segmaps = ia.SegmentationMapsOnImage(mask,shape=img.shape) 
    image_augmented, segmentation_map_augmented = seq(image=img,segmentation_maps=segmaps)
    visualizeImageAndMask(image_augmented,segmentation_map_augmented.get_arr())


