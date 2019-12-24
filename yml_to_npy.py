# -*- coding: utf-8 -*- 
'''
It's so slow for python to read yml files, but there are thouthands of such files. So for first step, yml need to be converted to numpy data.
'''

import numpy as np
from cv2 import cv2
import time
import os
import re

YMLFilePath = '/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data5'
# YMLFilePath = '/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data5'

ymlFiles = []
for root,dirs,files in os.walk(YMLFilePath):
    for f in files:
        if re.match('.*yml',f):
            ymlFile = os.path.join(root,f)
            ymlFiles.append(ymlFile)

datasize = len(ymlFiles)
t1 = time.time()

maxvalue = 0

for i,ymlFile in enumerate(ymlFiles):
    if i%100 == 0:
        print('Reading %d/%d, time = %d sec'%(i,datasize, (time.time() - t1)))
    fs = cv2.FileStorage(ymlFile,cv2.FileStorage_READ)
    if not fs.isOpened():
        print('Cannot open yml file, program exit')
        exit(-2)
    
    simulationTime = fs.getNode('simulationTime').real()/60 # minutes
    stateMap = fs.getNode('stateMap').mat()/simulationTime
    toRight = fs.getNode('toRight').mat()/simulationTime
    toLeft = fs.getNode('toLeft').mat()/simulationTime
    toUp = fs.getNode('toUp').mat()/simulationTime
    toDown = fs.getNode('toDown').mat()/simulationTime


    originPedestrianMatrix = fs.getNode('originPedestrianMatrix').mat()
    generatedPedestrianMatrix = fs.getNode('generatedPedestrianMatrix').mat()
    
    tup = (simulationTime,originPedestrianMatrix,generatedPedestrianMatrix,stateMap,toLeft,toRight,toUp,toDown)
    tup = np.array(tup)
    np.save(ymlFile[0:-4] + '.npy',tup)
    fs.release()