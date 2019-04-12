# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:02:17 2019

@author: Krishnasis
"""

from imageai.Detection import ObjectDetection
import os
from summarize import summarize

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


corpus=""
sentences=[]
image={}

lower_bound=0
upper_bound=50 #replace with the number of images to be processed

for i in range(lower_bound,upper_bound):
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "test2/image"+str(i)+".jpeg"), output_image_path=os.path.join(execution_path , "test2/imagenew"+str(i)+".jpg"))
    
    for eachObject in detections:
        corpus=corpus+ " "+eachObject["name"]
    
    sentences.append(corpus)
    image[corpus]="test2/image"+str(i)+".jpeg"


print(sentences)

summarize(image)
        