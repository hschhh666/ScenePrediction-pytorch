# -*- coding: utf-8 -*- 

'''
程序说明：东门和东南门各有自己的自编码器网络，开始实现算法
输入：均值状态地图
loss = E-E + S-S + E-S + S-E
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
from StateMapDataset import  FakeDataSet ,FakeAvgDataset
import os,sys
from logger import Logger
from AutoEncoder import BehaviorModelAutoEncoder
import itertools


if __name__ == '__main__':
    TestOrTrain = 'train'

    logfileName = 'log' + str(int(time.time()))+'.txt'
    sys.stdout = Logger(logfileName)
    
    # fakeEastGateTrainset = FakeDataSet('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = True)#训练集
    # fakeEastGateTestset = FakeDataSet('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate',train = False)#测试集

    # fakeEastSouthGateTrainset = FakeDataSet('/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate',train = True)#训练集
    # fakeEastSouthGateTestset = FakeDataSet('/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate',train = False)#测试集

    resultDir = '/home/hsc/Research/StateMapPrediction/code/models/EastAndSouth2/resultDir/'#可视化结果保存在这里
    fakeAvgTrainset = FakeAvgDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/AvgStateMap','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/AvgStateMap',train = True)
    fakeAvgTestset = FakeAvgDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/AvgStateMap','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/AvgStateMap',train = False)
    fakeAvgTrainLoader = DataLoader(fakeAvgTrainset,batch_size=4,shuffle=True)
    fakeAvgTestLoader = DataLoader(fakeAvgTestset,batch_size=4,shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device = ',device)

    if TestOrTrain =='train':

        EastModel = BehaviorModelAutoEncoder()
        SouthEastModel = BehaviorModelAutoEncoder()

        criterion = nn.MSELoss()
        # optimizer = optim.SGD(model.parameters(),lr = 0.0001,momentum=0.9)
        optimizer = optim.Adam(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.0001)

        EastModel.to(device)
        SouthEastModel.to(device)

        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()

        lastTestingLoss = np.inf

        for epoch in range(2000):#500个epoch
            running_loss = running_loss1 = running_loss2 = running_loss3 = running_loss4= running_zz = 0
            count = 0
            for i,sample in enumerate(fakeAvgTrainLoader):
                count += 1
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                Ez = EastModel.encoder(E)
                Sz = SouthEastModel.encoder(SE)
                EinEout = EastModel.decoder(Ez)
                EinSout = SouthEastModel.decoder(Ez)
                SinSout = SouthEastModel.decoder(Sz)
                SinEout = EastModel.decoder(Sz)

                loss1 = criterion(EinEout,E)
                loss2 = criterion(SinSout,SE)
                loss3 = criterion(EinSout,SE)
                loss4 = criterion(SinEout,E)
                zz_loss = criterion(Ez,Sz)

                loss = loss1 + loss2 + loss3 + loss4
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()
                running_loss4 += loss4.item()
                running_zz += zz_loss.item()

                if count == 10:
                    print('[%d, %5d] training loss: %.3f, E-E loss: %.3f, S-S loss: %.3f, E-S loss: %.3f, S-E loss: %.3f, z-z : %.3f' %(epoch + 1, i + 1, running_loss / count,running_loss1/count,running_loss2/count,running_loss3/count,running_loss4/count,running_zz/count))
                    count = 0
                    running_loss = running_loss1 = running_loss2 = running_loss3 = running_loss4= running_zz= 0
                    
            
            testing_loss = testing_loss1 = testing_loss2 = testing_loss3 = testing_loss4= testing_zz=0
            count = 0
            for i,sample in enumerate(fakeAvgTestLoader):
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                Ez = EastModel.encoder(E)
                Sz = SouthEastModel.encoder(SE)
                EinEout = EastModel.decoder(Ez)
                EinSout = SouthEastModel.decoder(Ez)
                SinSout = SouthEastModel.decoder(Sz)
                SinEout = EastModel.decoder(Sz)

                loss1 = criterion(EinEout,E)
                loss2 = criterion(SinSout,SE)
                loss3 = criterion(EinSout,SE)
                loss4 = criterion(SinEout,E)
                zz_loss = criterion(Ez,Sz)

                loss = loss1 + loss2 + loss3 + loss4  

                testing_loss  += loss.item()
                testing_loss1 += loss1.item()
                testing_loss2 += loss2.item()
                testing_loss3 += loss3.item()
                testing_loss4 += loss4.item()
                testing_zz += zz_loss.item()
                count += 1


                if i == 0:
                    concatenate = torch.cat([E,SE,EinEout,SinSout,SinEout,EinSout],0)
                    concatenate = concatenate.detach()
                    concatenate = concatenate.cpu()
                    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=True,pad_value=255)

                    concatenate = 255 - concatenate.numpy()*255
                    concatenate = np.transpose(concatenate,(1,2,0))
                    imgName = 'Epoch%d.jpg'%epoch
                    imgName = resultDir +imgName
                    cv2.imwrite(imgName,concatenate)
                    pass

            if testing_loss < lastTestingLoss:
                lastTestingLoss = testing_loss
                torch.save(EastModel.state_dict(), '/home/hsc/Research/StateMapPrediction/code/models/EastAndSouth2/modelParam/Easemodel.pth')
                torch.save(SouthEastModel.state_dict(), '/home/hsc/Research/StateMapPrediction/code/models/EastAndSouth2/modelParam/SEmodel.pth')

            print('[%d] testing loss: %.3f, E-E loss: %.3f, S-S loss: %.3f, E-S loss: %.3f, S-E loss: %.3f, z-z : %.3f' %(epoch + 1, testing_loss / count,testing_loss1/count,testing_loss2/count,testing_loss3/count,testing_loss4/count,testing_zz/count))
            print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
            using_time = time.time()-start_time
            hours = int(using_time/3600)
            using_time -= hours*3600
            minutes = int(using_time/60)
            using_time -= minutes*60
            print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)))

    if TestOrTrain == 'test':
        EastModel = BehaviorModelAutoEncoder()
        SouthEastModel = BehaviorModelAutoEncoder()

        EastModel.load_state_dict(torch.load('/home/hsc/Research/StateMapPrediction/code/models/EastAndSouth2/modelParam/Easemodel.pth'))
        SouthEastModel.load_state_dict(torch.load('/home/hsc/Research/StateMapPrediction/code/models/EastAndSouth2/modelParam/SEmodel.pth'))
        EastModel.to(device)
        SouthEastModel.to(device)

        for i,sample in enumerate(fakeAvgTrainLoader):
            E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)

            Ez = EastModel.encoder(E)
            EinSout = SouthEastModel.decoder(Ez)
            Sz = SouthEastModel.encoder(SE)
            SinEout = EastModel.decoder(Sz)

            concatenate = torch.cat([E,SE,SinEout,EinSout],0)
            concatenate = concatenate.detach()
            concatenate = concatenate.cpu()
            concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=True,pad_value=255)

            concatenate = 255 - concatenate.numpy()*255
            concatenate = np.transpose(concatenate,(1,2,0))
            imgName = '/home/hsc/testing.jpg'
            cv2.imwrite(imgName,concatenate)



            Ez = Ez.detach()
            Ez = Ez.cpu()
            Ez = Ez.numpy()
            Sz = Sz.detach()
            Sz = Sz.cpu()
            Sz = Sz.numpy()

            print(Ez[0,:])
            print(Sz[0,:])

            print('hhh')

            


            

        

        
