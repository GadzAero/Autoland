# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:03:04 2025

@author: pradi


Runway detection via Machine learning
"""
import os
import random
import shutil

import matplotlib.pyplot as plt
import cv2
import numpy as np




splitsize=0.85
categories=[]
source_folder="C:/Users/pradi/Downloads/archive"
folders=os.listdir(source_folder)
print (folders)

for subfolder in folders:
    if os.path.isdir(source_folder+"/"+subfolder):
        categories.append(subfolder)
categories.sort()
print(categories)

#Creer le target folder s'il n'existe pas 
target_folder= "C:/Users/pradi/Downloads/Finished_dataset/"
existDataSetPath= os.path.exists(target_folder)
if existDataSetPath==False:
    os.mkdir(target_folder)
    
def split_data(SOURCE,TRAINING, VALIDATION, SPLIT_SIZE):
    files=[]
    
    for filename in os.listdir(SOURCE):
        file=SOURCE+"/"+filename
        print(file)
        if os.path.getsize(file)>0:
            files.append(filename)
        else:
            print(filename + " ne contient pas de données, a ignorer")
    print(len(files))
    
split_data(source_folder + "/"+"1920x1080","","","")