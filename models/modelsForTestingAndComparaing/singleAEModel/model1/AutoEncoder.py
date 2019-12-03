# -*- coding: utf-8 -*- 

'''
程序说明：一个简单的小实验，说白了就是试试自编码器，看看自编码器恢复出来的图像能成啥样子
输入：状态地图中的行人占有栅格地图
输出：状态地图中的行人占有栅格地图
'''

import numpy as np
from cv2 import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
from datetime import datetime
from StateMapDataset import  FakeEastGateDataset
import os,sys
from logger import Logger

class BehaviorModelAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,3,3,padding=1)
        self.conv2 = nn.Conv2d(3,6,3,padding=1)
        self.conv3 = nn.Conv2d(6,9,3,padding=1)
        self.pool22 = nn.MaxPool2d(2,2)
        self.pool44 = nn.MaxPool2d(4,4)

        self.upsample22 = nn.Upsample(scale_factor=2,mode='nearest')
        self.conv4 = nn.Conv2d(9,6,3,padding=1)
        self.conv5 = nn.Conv2d(6,3,3,padding=1)
        self.conv6 = nn.Conv2d(3,1,3,padding=1)



    def encoder(self,x):
        x = self.pool22(F.relu(self.conv1(x)))
        x = self.pool22(F.relu(self.conv2(x)))
        x = self.pool22(F.relu(self.conv3(x)))
        return x

    def decoder(self,x):
        x = F.relu(self.conv4(self.upsample22(x)))
        x = F.relu(self.conv5(self.upsample22(x)))
        x = (self.conv6(self.upsample22(x)))
        return x


    def forward(self,x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_,z


if __name__ == '__main__':

    logfileName = 'log' + str(int(time.time()))+'.txt'
    sys.stdout = Logger(logfileName)

    trainORtest = 'train'
    print('this is ',trainORtest,' mode.')
    modelParamPATH = '/home/hsc/Research/StateMapPrediction/code/models/model1/modelParam/AEModelParam.pth'#模型参数保存在这里
    resultDir = '/home/hsc/Research/StateMapPrediction/code/models/model1/resultDir/'#可视化结果保存在这里
    fakeEastGateTrainset = FakeEastGateDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = True)#训练集
    fakeEastGateTestset = FakeEastGateDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = False)#测试集
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('device = ',device)


    if trainORtest =='fuck':
        model = BehaviorModelAutoEncoder()
        fakeEastGateTrainLoader = DataLoader(fakeEastGateTrainset,batch_size=4,shuffle=True)
        for i,sample in enumerate(fakeEastGateTrainLoader):
            a,b = sample['stateMap'], sample['pedestrianMatrix']
            z = model.encoder(a)
            pass


    if trainORtest == 'train':

        fakeEastGateTrainLoader = DataLoader(fakeEastGateTrainset,batch_size=4,shuffle=True)        
        fakeEastGateTestLoader = DataLoader(fakeEastGateTestset,batch_size=4,shuffle=False)


        model = BehaviorModelAutoEncoder()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(),lr = 0.001,momentum=0.9)
        model.to(device)

        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()

        for epoch in range(500):#500个epoch
            running_loss = 0
            for i,sample in enumerate(fakeEastGateTrainLoader):
                a,b = sample['stateMap'].to(device), sample['pedestrianMatrix'].to(device)
                optimizer.zero_grad()
                output,z = model(a)
                loss = criterion(output,a)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                if i%100 == 99:
                    print('[%d, %5d] training loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0
            
            
            torch.save(model.state_dict(), modelParamPATH)

            
            testing_loss = 0
            count = 0
            for i,sample in enumerate(fakeEastGateTestLoader):
                a,b = sample['stateMap'].to(device), sample['pedestrianMatrix'].to(device)
                optimizer.zero_grad()
                output,z = model(a)
                loss = criterion(output,a)        
                testing_loss  += loss.item()
                count += 1

                if i == 0:
                    concatenate = torch.cat([a,output],0)
                    concatenate = concatenate.detach()
                    concatenate = concatenate.cpu()
                    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=True,pad_value=255)

                    concatenate = 255 - concatenate.numpy()*255
                    concatenate = np.transpose(concatenate,(1,2,0))
                    imgName = 'Epoch%d.jpg'%epoch
                    imgName = resultDir +imgName
                    cv2.imwrite(imgName,concatenate)
                    pass

            print('[%d] testing loss: %.3f' %(epoch + 1,testing_loss/count))
            print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
            using_time = time.time()-start_time
            hours = int(using_time/3600)
            using_time -= hours*3600
            minutes = int(using_time/60)
            using_time -= minutes*60
            print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)))

    if trainORtest == 'test':
        # model = BehaviorModelAutoEncoder()
        # model.load_state_dict(torch.load(modelParamPATH))
        # torch.save(model,resultDir + 'model')
        mm = torch.load(resultDir + 'model')