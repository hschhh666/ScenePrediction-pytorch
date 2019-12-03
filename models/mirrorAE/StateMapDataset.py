# -*- coding: utf-8 -*- 

'''
读取状态地图的数据接口

'''



import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset
from cv2 import cv2
import random

class FakeDataSet(Dataset):
    def __init__(self,EastGateFakeDataPath,train = True):
        self.train = train
        self.npyFiles = []
        for root,_,files in os.walk(EastGateFakeDataPath):
            for f in files:
                if re.match('.*npy',f):
                    npyFile = os.path.join(root,f)
                    self.npyFiles.append(npyFile) #get all npy file in current directory
            break
        if len(self.npyFiles) ==0:
            print('Error! There is no npy file in directory ',EastGateFakeDataPath)
            print('Program exit.')
            exit(-1)
        else:
            print('There are %d npy files in directory'%len(self.npyFiles))
            random.seed(666) #shuffle npy files, but making sure every time has the same value
            random.shuffle(self.npyFiles)

    def __len__(self):
        if self.train:
            lenth = len(self.npyFiles)*0.8# 80% data as training set
        else:
            lenth = len(self.npyFiles)*0.2# 20% data as testing set
        return int(lenth)
        

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.train:
            idx = int(idx + len(self.npyFiles)*0.8)

        npyFile = self.npyFiles[idx]
        data = np.load(npyFile,allow_pickle=True)
        simulationTime = data[0]
        originPedestrianMatrix = data[1]
        generatedPedestrianMatrix = data[2]
        stateMap = data[3]
        stateMap = cv2.resize(stateMap,(512,512))
        stateMap = stateMap[np.newaxis,:]
        
        size = np.shape(generatedPedestrianMatrix)[0]
        pedestrianMatrix = np.zeros(size*size-size)

        count = 0
        for i in range(size):
            for j in range(size):
                if i==j:
                    continue
                pedestrianMatrix[count] = generatedPedestrianMatrix[i][j] #flatten matrix, but ignore diagonal element
                count = count + 1
        
        pedestrianMatrix = pedestrianMatrix.astype(np.float32)
        sample = {'stateMap':stateMap,'pedestrianMatrix':pedestrianMatrix}

        return sample


class FakeAvgDataset(Dataset):
    def __init__(self, E_path, SE_path, train = True):
        self.train = train
        self.E_path = E_path
        self.SE_path = SE_path

    def __len__(self):
        if self.train:
            return 80
        else:
            return 20

    def __getitem__(self,idx):
        idx += 1
        if not self.train:
            idx = idx + 80
        
        SE_npy = 'EastSouth_M' + str(idx)+'_AVG.npy'
        SE_npy = os.path.join(self.SE_path,SE_npy)
        E_npy = 'East_M' + str(idx) + '_AVG.npy'
        E_npy = os.path.join(self.E_path,E_npy)

        SEData = np.load(SE_npy,allow_pickle=True)
        EData = np.load(E_npy,allow_pickle=True)


        SEStateMap = SEData[3]
        SEStateMap = cv2.resize(SEStateMap,(512,512))
        SEStateMap = SEStateMap[np.newaxis,:]

        EStateMap = EData[3]
        EStateMap = cv2.resize(EStateMap,(512,512))
        EStateMap = EStateMap[np.newaxis,:]

        sample = {'EStateMap':EStateMap,'SEStateMap':SEStateMap}

        return sample




class FakeSinglePairDataset(Dataset):
    def __init__(self,E_path,SE_path,train = True):
        self.train = train
        self.E_path = E_path
        self.SE_path = SE_path
        self.Psize = 5
        self.Msize = 20

    def __len__(self):
        return self.Psize*self.Msize
    
    def __getitem__(self,idx):
        M = int(idx / self.Psize)
        M = M + 1
        P = random.randint(0,self.Psize-1)
        if not self.train:
            P = P+self.Psize
        SE_npy = 'EastSouth_M%d_P%d.npy'%(M,P)
        P = random.randint(0,self.Psize-1)
        if not self.train:
            P = P+self.Psize
        E_npy = 'East_M%d_P%d.npy'%(M,P)

        SE_npy = os.path.join(self.SE_path,SE_npy)
        E_npy = os.path.join(self.E_path,E_npy)

        SEData = np.load(SE_npy,allow_pickle=True)
        EData = np.load(E_npy,allow_pickle=True)

        SEStateMap = SEData[3]
        SEStateMap = cv2.resize(SEStateMap,(512,512))
        SEStateMap = SEStateMap[np.newaxis,:]

        EStateMap = EData[3]
        EStateMap = cv2.resize(EStateMap,(512,512))
        EStateMap = EStateMap[np.newaxis,:]

        sample = {'EStateMap':EStateMap,'SEStateMap':SEStateMap}

        return sample



if __name__ == '__main__':
    # test = FakeDataSet('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate')
    # sample = test[0]
    # a,b = sample['stateMap'], sample['pedestrianMatrix']
    # print('stateMap size = ',np.shape(a))
    # print('pedestrianMatrix size = ',np.shape(b))

    # maxValue = np.max(a)
    # a = 255 - (a/maxValue)*255
    # cv2.imwrite('/home/hsc/test.jpg',a[0,:,:])



    # test = FakeAvgDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/AvgStateMap','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/AvgStateMap')
    # sample = test[70]
    # E, SE = sample['EStateMap'], sample['SEStateMap']
    
    # maxValue = np.max(E)
    # E = 255 - (E/maxValue)*255
    # cv2.imwrite('/home/hsc/E.jpg',E[0,:,:])

    # maxValue = np.max(SE)
    # SE = 255 - (SE/maxValue)*255
    # cv2.imwrite('/home/hsc/SE.jpg',SE[0,:,:])

    test = FakeSinglePairDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data2','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data2')
    print(test[3])

    pass