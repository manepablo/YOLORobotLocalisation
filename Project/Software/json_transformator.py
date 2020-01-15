# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:32:30 2020

@author: Paul
"""
import os 
import json

size = 512
path = "C:\\Users\\Paul\\Desktop\\labels1"
outpath = "C:\\Users\\Paul\\Desktop\\labels"
y_data_list = os.listdir("C:\\Users\\Paul\\Desktop\\labels1")

for label in y_data_list:
    json_file = open(path + '\\' + label)
    oldjson = json.loads(json_file.read())       
    oldjsonbb3 = oldjson['objects'][0]['projected_cuboid']
    bb3d = []
    for  el in oldjson['objects'][0]['projected_cuboid']:
        bb3d.append(el[0]/size) 
        bb3d.append(el[1]/size)
    bb2d = []
    bb2d.append(oldjson['objects'][0]['bounding_box']['top_left'][1]/size)
    bb2d.append(oldjson['objects'][0]['bounding_box']['top_left'][0]/size)
    bb2d.append(oldjson['objects'][0]['bounding_box']['bottom_right'][1]/size)
    bb2d.append(oldjson['objects'][0]['bounding_box']['bottom_right'][0]/size)    
    outdic = {"class": "ESMRoboter", "centerPoint": [-1, -1], "3dBoundingBox": bb3d, "2dBoundingBox": bb2d}    
    with open(outpath + '\\' + label, 'w') as fp:
        json.dump(outdic, fp)
        