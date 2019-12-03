# -*- coding: utf-8 -*- 

'''
程序说明：东门和东南门各有自己的自编码器网络，开始实现算法，使用单次仿真数据而非平均数据
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
from StateMapDataset import  FakeDataSet ,FakeAvgDataset, FakeSinglePairDataset
import os,sys
from logger import Logger
from AutoEncoder import BehaviorModelAutoEncoder
import itertools


if __name__ == '__main__':

    TestOrTrain = 'train'
    saveThisExper = False

    E_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data3'
    SE_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data3'

    # 当前路径下的resultDir用来保存每次的试验结果，包括log、结果图、训练参数。每次实验都在resultDir下创建一个以实验开始时间为名字的文件夹，该文件夹下保存当次实验的所有结果。
    # 如果resultDir不存在，则创建
    curPath = os.path.split(os.path.realpath(__file__))[0]
    resultDir = 'resultDir'
    resultDir = os.path.join(curPath,resultDir)
    if not os.path.exists(resultDir):
        print('create result dir')
        os.makedirs(resultDir)
    # 获取实验开始时间，并在resultDir下创建以该时间为名字的文件夹，用以保存本次实验结果
    curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
    if saveThisExper:
        resultDir = os.path.join(resultDir,curTime)
        os.makedirs(resultDir)
    else:
       resultDir = os.path.join(resultDir,'tmp')
       resultDir = os.path.join(resultDir,curTime)
       os.makedirs(resultDir)

    

    # 创建log文件、img文件夹和modelParam文件夹，分别表示本次实验的日志、实验结果存储文件夹和模型参数存储文件夹
    logfileName = os.path.join(resultDir,curTime+'.txt')
    sys.stdout = Logger(logfileName)
    imgFolder = os.path.join(resultDir,'img')
    os.makedirs(imgFolder)
    modelParamFolder = os.path.join(resultDir,'modelParam')
    os.makedirs(modelParamFolder)

    # 加载数据集
    fakeSingleTrainset = FakeSinglePairDataset(E_dataset_path,SE_dataset_path,train = True)
    fakeSingleTestset = FakeSinglePairDataset(E_dataset_path,SE_dataset_path,train = False)
    fakeSingleTrainLoader = DataLoader(fakeSingleTrainset,batch_size=4,shuffle=True)
    fakeSingleTestLoader = DataLoader(fakeSingleTestset,batch_size=4,shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device = ',device)

    if TestOrTrain =='train':

        EastModel = BehaviorModelAutoEncoder()
        SouthEastModel = BehaviorModelAutoEncoder()
        theta1 = torch.Tensor([1])
        theta2 = torch.Tensor([0.1])
        theta3 = torch.Tensor([10])

        EastModel.to(device)
        SouthEastModel.to(device)
        theta1 = theta1.cuda()
        theta2 = theta2.cuda()
        theta3 = theta3.cuda()
        
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.001,momentum=0.9)
        # optimizer = optim.Adam(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.001)
        optimizer = optim.Adam([{'params':EastModel.parameters()},{'params':SouthEastModel.parameters()},{'params':theta1},{'params':theta2},{'params':theta3,'lr':0.01}],lr = 0.001)
        
        theta1.requires_grad = True
        theta2.requires_grad = True
        theta3.requires_grad = True


        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()
        lastTestingLoss = np.inf
        for epoch in range(2000):#500个epoch
            print('theta1 = %.3f, theta2 = %.3f, theta3 = %.3f'%(theta1.item(),theta2.item(),theta3.item()))
            running_loss = running_loss1 = running_loss2 = running_loss3 = 0
            count = 0
            for i,sample in enumerate(fakeSingleTrainLoader):
                count += 1
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                EOut,Ez = EastModel(E)
                SOut,Sz = SouthEastModel(SE)

                loss1 = criterion(EOut,E)
                loss2 = criterion(SOut,SE)
                loss3 = criterion(Ez,Sz)

                loss = loss1/theta1 +  loss2/theta2 + loss3/theta3 + torch.log(theta1*theta1) + torch.log(theta2*theta2) + torch.log(theta3*theta3)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                running_loss1 += loss1.item()
                running_loss2 += loss2.item()
                running_loss3 += loss3.item()

                if count == 10:
                    print('[%d, %5d] training loss: %.3f, E-E recons loss: %.3f, S-S recons loss: %.3f, z-z loss: %.3f' %(epoch + 1, i + 1, running_loss / count,running_loss1/count,running_loss2/count,running_loss3/count))
                    count = 0
                    running_loss = running_loss1 = running_loss2 = running_loss3 = 0
                    
                        
            testing_loss = testing_loss1 = testing_loss2 = testing_loss3 = 0
            count = 0
            for i,sample in enumerate(fakeSingleTestLoader):
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                EOut,Ez = EastModel(E)
                SOut,Sz = SouthEastModel(SE)

                loss1 = criterion(EOut,E)
                loss2 = criterion(SOut,SE)
                loss3 = criterion(Ez,Sz)
                loss = loss1 + loss2 + loss3     

                testing_loss  += loss.item()
                testing_loss1 += loss1.item()
                testing_loss2 += loss2.item()
                testing_loss3 += loss3.item()
                count += 1

                EinSout = SouthEastModel.decoder(Ez)
                SinEout = EastModel.decoder(Sz)

                if i == 0:
                    concatenate = torch.cat([E,SE,EOut,SOut,SinEout,EinSout],0)
                    concatenate = concatenate.detach()
                    concatenate = concatenate.cpu()
                    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=True,pad_value=255)

                    concatenate = 255 - concatenate.numpy()*255
                    concatenate = np.transpose(concatenate,(1,2,0))
                    imgName = 'Test_Epoch%d.jpg'%epoch
                    imgName = os.path.join(imgFolder,imgName)
                    cv2.imwrite(imgName,concatenate)
                    pass

            for i,sample in enumerate(fakeSingleTrainLoader):
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                EOut,Ez = EastModel(E)
                SOut,Sz = SouthEastModel(SE)

                loss1 = criterion(EOut,E)
                loss2 = criterion(SOut,SE)
                loss3 = criterion(Ez,Sz)
                loss = loss1 + loss2 + loss3     

                testing_loss  += loss.item()
                testing_loss1 += loss1.item()
                testing_loss2 += loss2.item()
                testing_loss3 += loss3.item()
                count += 1

                EinSout = SouthEastModel.decoder(Ez)
                SinEout = EastModel.decoder(Sz)

                if i == 0:
                    concatenate = torch.cat([E,SE,EOut,SOut,SinEout,EinSout],0)
                    concatenate = concatenate.detach()
                    concatenate = concatenate.cpu()
                    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=True,pad_value=255)

                    concatenate = 255 - concatenate.numpy()*255
                    concatenate = np.transpose(concatenate,(1,2,0))
                    imgName = 'Train_Epoch%d.jpg'%epoch
                    imgName = os.path.join(imgFolder,imgName)
                    cv2.imwrite(imgName,concatenate)
                    pass

            # if  epoch > 100 and epoch%2 == 0 and testing_loss < lastTestingLoss:
            if testing_loss < lastTestingLoss:
                lastTestingLoss = testing_loss
                torch.save(EastModel.state_dict(),os.path.join(modelParamFolder,'Easemodel.pth'))
                torch.save(SouthEastModel.state_dict(),os.path.join(modelParamFolder,'SEmodel.pth'))

            print('[%d] testing loss: %.3f, E-E recons loss: %.3f, S-S recons loss: %.3f, z-z loss: %.3f' %(epoch + 1, testing_loss / count,testing_loss1/count,testing_loss2/count,testing_loss3/count))
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

        EastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'Easemodel.pth')))
        SouthEastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'SEmodel.pth')))
        EastModel.to(device)
        SouthEastModel.to(device)
        criterion = nn.MSELoss()

        for i,sample in enumerate(fakeSingleTrainLoader):
            E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)

            Ez = EastModel.encoder(E)
            EinSout = SouthEastModel.decoder(Ez)
            EinEout = EastModel.decoder(Ez)

            Sz = SouthEastModel.encoder(SE)
            SinEout = EastModel.decoder(Sz)
            SinSout = SouthEastModel.decoder(Sz)

            concatenate = torch.cat([E,SE,EinEout,SinSout, SinEout,EinSout],0)
            concatenate = concatenate.detach()
            concatenate = concatenate.cpu()
            concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=True,pad_value=255)

            concatenate = 255 - concatenate.numpy()*255
            concatenate = np.transpose(concatenate,(1,2,0))
            imgName = '/home/hsc/testing.jpg'
            cv2.imwrite(imgName,concatenate)

            
            loss = criterion(Ez,Sz)


            Ez = Ez.detach()
            Ez = Ez.cpu()
            Ez = Ez.numpy()
            Sz = Sz.detach()
            Sz = Sz.cpu()
            Sz = Sz.numpy()

            print(Ez[0,:])
            print(Sz[0,:])
            print(loss.item())

            print('hhh')

            


            

        

        
