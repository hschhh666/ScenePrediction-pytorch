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


        self.ins3 = nn.InstanceNorm2d(3)
        self.ins6 = nn.InstanceNorm2d(6)
        self.ins9 = nn.InstanceNorm2d(9)
        self.ins12 = nn.InstanceNorm2d(12)
        self.ins15 = nn.InstanceNorm2d(15)
        self.ins18 = nn.InstanceNorm2d(18)
        self.ins20 = nn.InstanceNorm2d(20)

        self.e_conv1 = nn.Conv2d(1,3,3,padding=1,stride=2)
        self.e_conv2 = nn.Conv2d(3,6,3,padding=1,stride=2)
        self.e_conv3 = nn.Conv2d(6,9,3,padding=1,stride=2)
        self.e_conv4 = nn.Conv2d(9,12,3,padding=1,stride=2)
        self.e_conv5 = nn.Conv2d(12,15,3,padding=1,stride=2)
        self.e_conv6 = nn.Conv2d(15,18,3,padding=1,stride=2)
        self.e_conv7 = nn.Conv2d(18,18,3,padding=1,stride=1)
        self.e_conv8 = nn.Conv2d(18,18,3,padding=1,stride=1)
        self.e_conv9 = nn.Conv2d(18,20,3,padding=1,stride=2)
        self.e_conv10 = nn.Conv2d(20,20,4,padding=0,stride=1)

        self.d_conv1 = nn.ConvTranspose2d(20,20,4)
        self.d_conv2 = nn.ConvTranspose2d(20,18,4,padding = 1, stride = 2)
        self.d_conv3 = nn.ConvTranspose2d(18,18,3,padding = 1, stride = 1)
        self.d_conv4 = nn.ConvTranspose2d(18,18,3,padding = 1, stride = 1)
        self.d_conv5 = nn.ConvTranspose2d(18,15,4,padding = 1, stride = 2)
        self.d_conv6 = nn.ConvTranspose2d(15,12,4,padding = 1, stride = 2)
        self.d_conv7 = nn.ConvTranspose2d(12,9,4,padding = 1, stride = 2)
        self.d_conv8 = nn.ConvTranspose2d(9,6,4,padding = 1, stride = 2)
        self.d_conv9 = nn.ConvTranspose2d(6,3,4,padding = 1, stride = 2)
        self.d_conv10 = nn.ConvTranspose2d(3,1,4,padding = 1, stride = 2)



    def encoder(self,x):
        x = F.leaky_relu(self.e_conv1(x))
        x = self.ins6(F.relu(self.e_conv2(x)))
        x = self.ins9(F.relu(self.e_conv3(x)))
        x = self.ins12(F.relu(self.e_conv4(x)))
        x = self.ins15(F.relu(self.e_conv5(x)))
        x = self.ins18(F.relu(self.e_conv6(x)))
        res1 = x
        x = self.ins18(F.relu(self.e_conv7(x)))
        x = self.ins18(F.relu(self.e_conv8(x)))
        x += res1
        x = self.ins20(F.relu(self.e_conv9(x)))
        x = (F.relu(self.e_conv10(x)))
        return x

    def decoder(self,x):
        x = F.relu(self.d_conv1(x))
        x = self.ins18(F.relu(self.d_conv2(x)))
        res2 = x
        x = self.ins18(F.relu(self.d_conv3(x)))
        x = self.ins18(F.relu(self.d_conv4(x)))
        x += res2
        x = self.ins15(F.relu(self.d_conv5(x)))
        x = self.ins12(F.relu(self.d_conv6(x)))
        x = self.ins9(F.relu(self.d_conv7(x)))
        x = self.ins6(F.relu(self.d_conv8(x)))
        x = self.ins3(F.relu(self.d_conv9(x)))
        x = F.leaky_relu((self.d_conv10(x)))
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
    modelParamPATH = '/home/hsc/Research/StateMapPrediction/code/models/model5/modelParam/AEModelParam.pth'#模型参数保存在这里
    resultDir = '/home/hsc/Research/StateMapPrediction/code/models/model5/resultDir/'#可视化结果保存在这里
    fakeEastGateTrainset = FakeEastGateDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = True)#训练集
    fakeEastGateTestset = FakeEastGateDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = False)#测试集
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device = ',device)


    if trainORtest =='fuck':
        model = BehaviorModelAutoEncoder()
        fakeEastGateTrainLoader = DataLoader(fakeEastGateTrainset,batch_size=4,shuffle=True)
        for i,sample in enumerate(fakeEastGateTrainLoader):
            a,b = sample['stateMap'], sample['pedestrianMatrix']
            x_,z = model.forward(a)
            pass


    if trainORtest == 'train':
        lastTestLoss = np.inf

        fakeEastGateTrainLoader = DataLoader(fakeEastGateTrainset,batch_size=4,shuffle=True)        
        fakeEastGateTestLoader = DataLoader(fakeEastGateTestset,batch_size=4,shuffle=False)


        model = BehaviorModelAutoEncoder()
        print(model)
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(),lr = 0.0001,momentum=0.9)
        optimizer = optim.Adam(model.parameters(),lr = 0.0001)

        model.to(device)

        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()

        for epoch in range(1000):#500个epoch
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

            if (testing_loss/count) < lastTestLoss:
                torch.save(model.state_dict(), modelParamPATH)
                lastTestLoss = (testing_loss/count)

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