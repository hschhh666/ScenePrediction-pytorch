# -*- coding: utf-8 -*- 

'''
程序说明：一个简单的小实验，为了验证在给定行为模式数量的前提下，网络是否具备识别每种行为模式数量的能力。结论是可以的
输入：状态地图中的行人占有栅格地图
输出：n维向量，n大小等于行为模式的个数，每一维度表示当前输入下这种行为模式上有多少人
误差为MSE，训练集误差可以做到0.001级别，测试集误差在0.8附近，可以认为网络具备学习行为模式上数量的能力。
'''
import numpy as np
from cv2 import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime
from StateMapDataset import  FakeEastGateDataset


class BehaviorModelNet(nn.Module):# 网络模型定位输出为12维，因为东门共4个拓扑点，所以行为模式有4x4-4=12种
    def __init__(self):
        super(BehaviorModelNet,self).__init__()
        self.conv1 = nn.Conv2d(1,3,5)
        self.conv2 = nn.Conv2d(3,6,5)
        self.conv3 = nn.Conv2d(6,9,5)
        self.pool22 = nn.MaxPool2d(2,2)
        self.pool44 = nn.MaxPool2d(4,4)
        self.fc1 = nn.Linear(14*14*9,882)
        self.fc2 = nn.Linear(882,84)
        self.fc3 = nn.Linear(84,12)
        pass

    def forward(self,x):
        x = self.pool22(F.relu(self.conv1(x)))
        x = self.pool44(F.relu(self.conv2(x)))
        x = self.pool44(F.relu(self.conv3(x)))
        x = x.view(-1,14*14*9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':

    trainORtest = 'test'
    modelParamPATH = '/home/hsc/Research/StateMapPrediction/code/networkParam/modelParam.pth'#模型参数保存在这里
    fakeEastGateTrainset = FakeEastGateDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = True)#训练集
    fakeEastGateTestset = FakeEastGateDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = False)#测试集
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if trainORtest == 'train':

        fakeEastGateTrainLoader = DataLoader(fakeEastGateTrainset,batch_size=4,shuffle=True)        
        fakeEastGateTestLoader = DataLoader(fakeEastGateTestset,batch_size=16,shuffle=False)

        # for i,sample in enumerate(fakeEastGateTestLoader):
        #     a,b = sample['stateMap'], sample['pedestrianMatrix']
        #     a = a[0].numpy()
        #     maxValue = np.max(a)
        #     a = 255 - (a/maxValue)*255
        #     cv2.imwrite('/home/hsc/test.jpg',a[0,:,:])
        #     print(i)

        model = BehaviorModelNet()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(),lr = 0.001,momentum=0.9)
        model.to(device)


        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()

        for epoch in range(100):#100个epoch
            running_loss = 0
            for i,sample in enumerate(fakeEastGateTrainLoader):
                a,b = sample['stateMap'].to(device), sample['pedestrianMatrix'].to(device)
                optimizer.zero_grad()
                output = model(a)
                loss = criterion(output,b)
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
                output = model(a)
                loss = criterion(output,b)        
                testing_loss  += loss.item()
                count += 1
            print('[%d] testing loss: %.3f' %(epoch + 1,testing_loss/count),end = ' ')
            print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
            using_time = time.time()-start_time
            hours = int(using_time/3600)
            using_time -= hours*3600
            minutes = int(using_time/60)
            using_time -= minutes*60
            print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)))

    if trainORtest == 'test':
        model = BehaviorModelNet()
        model.load_state_dict(torch.load(modelParamPATH))
        model.to(device)
        criterion = nn.MSELoss()


        fakeEastGateTestLoader = DataLoader(fakeEastGateTestset,batch_size=1,shuffle=False)

        for i,sample in enumerate(fakeEastGateTestLoader):
            a,b = sample['stateMap'].to(device), sample['pedestrianMatrix'].to(device)
            output = model(a)
            loss = criterion(output,b)
            output = output.cpu()
            output = output.detach()

            b = b.cpu()

            output = output.numpy()
            b = b.numpy()

            for i in range(12):
                print('%.2f, %.2f'%(output[0,i],b[0,i]))#打印出来每张状态地图的预测结果和真值
            
            print('loss = %.5f'%loss)
            print('*****************\n*****************')
            input()