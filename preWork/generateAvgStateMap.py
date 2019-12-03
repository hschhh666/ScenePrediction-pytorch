import numpy as np
import os,sys
import re
from cv2 import cv2

singleNpyFilePath = '/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate'
avgNpyFilePath = os.path.join(singleNpyFilePath,'AvgStateMap')

singleNpyFiles = []
for root,_,files in os.walk(singleNpyFilePath):
    for f in files:
        match = re.match('.*npy',f)
        if match!=None:
            singleNpyFiles.append(f)
    break

if len(singleNpyFiles) == 0:
    print('no npy files, program exit')
    exit(-1)


baseName = ''
M_max = -np.inf
M_min = np.inf
P_max = -np.inf
P_min = np.inf
for f in singleNpyFiles:
    match = re.match('(.*)_M(\d*)_P(\d*)\.npy',f)
    if match != None:
        baseName = match.group(1)
        M = int(match.group(2))
        P = int(match.group(3))
        if M > M_max:
            M_max = M
        if M < M_min:
            M_min = M
        if P > P_max:
            P_max = P
        if P < P_min:
            P_min = P

print(M_max,M_min,P_max,P_min)

simulationTime = ''
originPedestrianMatrix = ''
avgGeneratedPedestrianMatrix = ''
avgStateMap = ''
for m in range(M_min,M_max+1):
    count = 0
    for p in range(P_min,P_max+1):
        singleNpyFile = baseName + '_M' + str(m) + '_P' + str(p) + '.npy'
        singleNpyFile = os.path.join(singleNpyFilePath,singleNpyFile)
        data = np.load(singleNpyFile,allow_pickle=True)
        if count == 0:
            simulationTime = data[0]
            originPedestrianMatrix = data[1]
            avgGeneratedPedestrianMatrix = data[2]
            avgStateMap = data[3]
        else:
            simulationTime += data[0]
            originPedestrianMatrix += data[1]
            avgGeneratedPedestrianMatrix += data[2]
            avgStateMap += data[3] 

        count += 1

    simulationTime /= count
    originPedestrianMatrix /= count
    avgGeneratedPedestrianMatrix /= count
    avgStateMap /= count

    tup = (simulationTime,originPedestrianMatrix,avgGeneratedPedestrianMatrix,avgStateMap)
    tup = np.array(tup)
    avgNpyFile = baseName + '_M' + str(m) + '_AVG.npy'
    avgJpgFile = baseName + '_M' + str(m) + '_AVG.jpg'
    avgNpyFile = os.path.join(avgNpyFilePath,avgNpyFile)
    avgJpgFile = os.path.join(avgNpyFilePath,avgJpgFile)
    
    np.save(avgNpyFile,tup)

    mmax = np.max(avgStateMap)
    avgStateMap/=mmax
    avgStateMap *=255
    avgStateMap = 255 - avgStateMap
    cv2.imwrite(avgJpgFile,avgStateMap)
    


        
        
        