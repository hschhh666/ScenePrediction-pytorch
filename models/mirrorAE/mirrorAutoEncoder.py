# -*- coding: utf-8 -*- 

'''
程序说明：东门和东南门各有自己的自编码器网络，开始实现算法，使用单次仿真数据而非平均数据
'''

import numpy as np
from cv2 import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
from datetime import datetime
from StateMapDataset import  FakeDeltaTDataset, typicalTestData, convertDataToBGR
import os,sys
from logger import Logger
from AutoEncoder import BehaviorModelAutoEncoder
import itertools
from tensorboardX import SummaryWriter
import argparse


if __name__ == '__main__':


    argParser = argparse.ArgumentParser(description='python arguments')
    argParser.add_argument('-cuda',type=int ,help='cuda device id')
    argParser.add_argument('-zdim',type=int,help='z dimention')
    argParser.add_argument('-dropout',type=float ,help='dropout p',default=0)
    args = argParser.parse_args()
    if args.cuda == None or args.zdim == None:
        print('[Error] No parameter. Program exit')
        exit(-2)
    if args.cuda < 0 or args.cuda >= torch.cuda.device_count():
        print('cuda %d does not exit! Program exit'%args.cuda)
        exit(-2)
    if args.zdim <= 0:
        print('z dim cannot <= zero! Program exit')
        exit(-2)
    if args.dropout >=1 or args.dropout <0:
        print('dropout p should be [0,1). Program exit')
        exit(-2)

    TestOrTrain = 'train'
    saveThisExper = False

    E_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data5'
    SE_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data5'

    device = 'cuda:' + str(args.cuda)
    device = torch.device(device)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if TestOrTrain =='train':

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

        tbLogDir = os.path.join(resultDir,'tbLog')
        tensorboardWriter = SummaryWriter(logdir=tbLogDir,flush_secs=1)

        # 创建log文件、img文件夹和modelParam文件夹，分别表示本次实验的日志、实验结果存储文件夹和模型参数存储文件夹
        logfileName = os.path.join(resultDir,curTime+'.txt')
        sys.stdout = Logger(logfileName)
        imgFolder = os.path.join(resultDir,'img')
        os.makedirs(imgFolder)
        modelParamFolder = os.path.join(resultDir,'modelParam')
        os.makedirs(modelParamFolder)

        # 加载数据集
        fakeSingleTrainset = FakeDeltaTDataset(E_dataset_path,SE_dataset_path,1,train = True)
        fakeSingleTrainLoader = DataLoader(fakeSingleTrainset,batch_size=4,shuffle=True)

        fakeSingleTrainsets = [FakeDeltaTDataset(E_dataset_path,SE_dataset_path,i,train = True) for i in range(5)]
        fakeSingleTrainLoaders = [DataLoader(fakeSingleTrainsets[i],batch_size=4,shuffle=True)  for i in range(5)]

        fakeSingleTestset = FakeDeltaTDataset(E_dataset_path,SE_dataset_path,0,train = False)
        fakeSingleTestLoader = DataLoader(fakeSingleTestset,batch_size=4,shuffle=True)
        
        print('device = ',device)
        print('z-dim = ',args.zdim)
        print('dropout = ',args.dropout)

        # 加载模型
        EastModel = BehaviorModelAutoEncoder(args.zdim , args.dropout)
        SouthEastModel = BehaviorModelAutoEncoder(args.zdim , args.dropout)
        theta1 = torch.Tensor([1])
        theta2 = torch.Tensor([0.1])
        theta3 = torch.Tensor([10])

        # 模型迁移到GPU
        EastModel.to(device)
        SouthEastModel.to(device)
        theta1 = theta1.cuda(device = device)
        theta2 = theta2.cuda(device = device)
        theta3 = theta3.cuda(device = device)

        
        
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.001,momentum=0.9)
        # optimizer = optim.Adam(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.001)
        optimizer = optim.Adam([{'params':EastModel.parameters()},{'params':SouthEastModel.parameters()},{'params':theta1,'lr':0.01},{'params':theta2,'lr':0.01},{'params':theta3,'lr':0.01}],lr = 0.001)
        
        theta1.requires_grad = True
        theta2.requires_grad = True
        theta3.requires_grad = True


        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()
        lastTestingLoss = np.inf
        minPredictionLoss = np.inf

        # 2000个epoch
        for epoch in range(2000):

            # 每个epoch都将需要计算的内容归零
            running_loss = running_loss1 = running_loss2 = running_loss3 = 0
            count = 0

            # 可视化训练集上的loss
            lossList = []
            # 可视化训练集上的e-e和s-s重构误差
            reconsDict = {}
            # 可视化不同deltaT的隐变量距离
            zzDict = {}
            # 可视化网络中不同层的平均梯度值以检查是否有梯度消失现象
            gradDict = {}
            # 训练
            EastModel.train()
            SouthEastModel.train()
            for i in range(5):
                fakeSingleTrainLoader = fakeSingleTrainLoaders[i]
                count = 0
                for i,sample in enumerate(fakeSingleTrainLoader):
                    trainingPercent = int(100 * (i+1)/fakeSingleTrainLoader.__len__())
                    count += 1
                    E,SE,deltaT = sample['EStateMap'].to(device), sample['SEStateMap'].to(device),sample['deltaT']
                    deltaT = int(deltaT[0])
                    optimizer.zero_grad()

                    EOut,Ez = EastModel(E)
                    SOut,Sz = SouthEastModel(SE)

                    loss1 = criterion(EOut,E)
                    loss2 = criterion(SOut,SE)
                    loss3 = criterion(Ez,Sz)

                    # coefficent = (1.0/(deltaT + 1.0))
                    coefficent = np.exp(-0.55*deltaT)
                    coefficent = float(coefficent)

                    # loss = loss1/theta1 +  loss2/theta2 + coefficent * loss3/theta3 + torch.log(theta1*theta1) + torch.log(theta2*theta2) + torch.log(theta3*theta3)

                    loss = loss1 + loss2 + loss3 * 1 * coefficent

                    loss.backward()
                    optimizer.step()

                    # 可视化网络深层、中层、浅层的平均梯度到tensorboard
                    paramsDict = dict(EastModel.named_parameters())
                    gradDict.setdefault('conv1_weight',[])
                    gradDict['conv1_weight'].append(np.average((paramsDict['conv1.weight'].grad).detach().cpu().numpy()))
                    gradDict.setdefault('fc2_weight',[])
                    gradDict['fc2_weight'].append(np.average((paramsDict['fc2.weight'].grad).detach().cpu().numpy()))
                    gradDict.setdefault('dconv3_weight',[])
                    gradDict['dconv3_weight'].append(np.average((paramsDict['dconv3.weight'].grad).detach().cpu().numpy()))

                    running_loss += loss.item()
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()
                    running_loss3 += loss3.item()

                    # 可视化loss、reconsLoss和deltaT z-z loss到tensorboard
                    lossList.append(loss.item())
                    reconsDict.setdefault('ee',[])
                    reconsDict['ee'].append(loss1.item())
                    reconsDict.setdefault('ss',[])
                    reconsDict['ss'].append(loss2.item())
                    nowDeltaT = 'deltaT' + str(deltaT)
                    zzDict.setdefault(nowDeltaT,[])
                    zzDict[nowDeltaT].append(loss3.item())

                    if count == 1:
                        if fakeSingleTrainLoader.__len__() - (i+1) < count:
                            trainingPercent = 100
                        print('[%d, %5d%%]deltaT = %d, training loss: %.3f, E-E recons loss: %.3f, S-S recons loss: %.3f, z-z loss: %.5f' %(epoch + 1, trainingPercent, deltaT,running_loss / count,running_loss1/count,running_loss2/count,running_loss3/count))
                        count = 0
                        running_loss = running_loss1 = running_loss2 = running_loss3 = 0

            # 计算需要可视化的内容的均值
            running_loss3 = np.average(list(zzDict.values()))
            for key in reconsDict.keys():
                reconsDict[key] = np.average(reconsDict[key])
            for key in zzDict.keys():
                zzDict[key] = np.average(zzDict[key])
            running_loss = np.average(lossList)
            running_loss1 = reconsDict['ee']
            running_loss2 = reconsDict['ss']
            for key in gradDict.keys():
                gradDict[key] = np.average(gradDict[key])
            
            # 可视化到tensorboard
            tensorboardWriter.add_scalar('training/loss',running_loss,epoch)
            tensorboardWriter.add_scalars('training/recons loss',{'E-E recons loss':running_loss1,'S-S recons loss':running_loss2},epoch)
            tensorboardWriter.add_scalar('training/z-z loss',running_loss3,epoch)
            tensorboardWriter.add_scalars('training/z-z single deltaT loss',zzDict,epoch)
            tensorboardWriter.add_scalars('training/grads',gradDict,epoch)
            tensorboardWriter.flush()

            # 以下为测试时的内容
            # 每个epoch都将需要计算的内容归零
            EPredictionLoss = SEPredictionLoss = predictionLoss = testing_loss = testing_loss1 = testing_loss2 = testing_loss3 = 0
            count = 0
            
            # 计算当前epoch的testing loss，并可视化部分testing结果
            EastModel.eval()
            SouthEastModel.eval()
            for i,sample in enumerate(fakeSingleTestLoader):
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                EOut,Ez = EastModel(E)
                SOut,Sz = SouthEastModel(SE)
                EinSout = SouthEastModel.decoder(Ez)
                SinEout = EastModel.decoder(Sz)

                loss1 = criterion(EOut,E)
                loss2 = criterion(SOut,SE)
                loss3 = criterion(Ez,Sz)

                # 东门、东南门和俩门的预测误差
                EPredictionLoss += criterion(SinEout,E)
                SEPredictionLoss += criterion(EinSout,SE)
                predictionLoss += ((criterion(SinEout,E) + criterion(EinSout,SE))/2).item()

                # loss = loss1/theta1 +  loss2/theta2 + loss3/theta3 + torch.log(theta1*theta1) + torch.log(theta2*theta2) + torch.log(theta3*theta3)     

                loss = loss1 + loss2

                testing_loss  += loss.item()
                testing_loss1 += loss1.item()
                testing_loss2 += loss2.item()
                testing_loss3 += loss3.item()
                count += 1

                
                if i == 0:
                    concatenate = torch.cat([E,SE,EOut,SOut,SinEout,EinSout],0)
                    concatenate = concatenate.detach()
                    concatenate = concatenate.cpu()
                    concatenate = convertDataToBGR(concatenate)
                    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=False,pad_value=0)

                    concatenate = concatenate.numpy()
                    concatenate = np.transpose(concatenate,(1,2,0))
                    imgName = 'Test_Epoch%d.jpg'%epoch
                    imgName = os.path.join(imgFolder,imgName)
                    cv2.imwrite(imgName,concatenate)
            
            # 可视化在训练集上的部分结果
            for i,sample in enumerate(fakeSingleTrainLoader):
                E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
                optimizer.zero_grad()

                EOut,Ez = EastModel(E)
                SOut,Sz = SouthEastModel(SE)
                EinSout = SouthEastModel.decoder(Ez)
                SinEout = EastModel.decoder(Sz)

                concatenate = torch.cat([E,SE,EOut,SOut,SinEout,EinSout],0)
                concatenate = concatenate.detach()
                concatenate = concatenate.cpu()
                concatenate = convertDataToBGR(concatenate)
                concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=False,pad_value=0)

                concatenate = concatenate.numpy()
                concatenate = np.transpose(concatenate,(1,2,0))
                imgName = 'Train_Epoch%d.jpg'%epoch
                imgName = os.path.join(imgFolder,imgName)
                cv2.imwrite(imgName,concatenate)
                break
            
            # 保存有史以来predictionLoss最小的网络参数
            if predictionLoss < minPredictionLoss:
                minPredictionLoss = predictionLoss
                torch.save(EastModel.state_dict(),os.path.join(modelParamFolder,'Easemodel.pth'))
                torch.save(SouthEastModel.state_dict(),os.path.join(modelParamFolder,'SEmodel.pth'))

            if testing_loss < lastTestingLoss:
                lastTestingLoss = testing_loss
                torch.save(EastModel.state_dict(),os.path.join(modelParamFolder,'Easemodel_testloss.pth'))
                torch.save(SouthEastModel.state_dict(),os.path.join(modelParamFolder,'SEmodel_testloss.pth'))

            print()
            print('[%d，%6s] testing  loss: %.3f, prediction loss: %.3f, E-E recons loss: %.3f, S-S recons loss: %.3f, z-z loss: %.5f' %(epoch + 1,'--', testing_loss / count,predictionLoss/count,testing_loss1/count,testing_loss2/count,testing_loss3/count))
            print('[%d, %6s] theta1 = %.3f, theta2 = %.3f, theta3 = %.3f'%(epoch+1, '--',theta1.item(),theta2.item(),theta3.item()))

            # 可视化testing loss、recons loss、俩门和各自门的prediction loss到tensorboard
            tensorboardWriter.add_scalar('testing/loss',testing_loss/count,epoch)
            tensorboardWriter.add_scalars('testing/recons loss',{'E-E recons loss':testing_loss1/count,'S-S recons loss':testing_loss2/count},epoch)
            tensorboardWriter.add_scalar('testing/z-z loss',testing_loss3/count,epoch)
            tensorboardWriter.add_scalar('testing/prediction loss',predictionLoss/count,epoch)
            tensorboardWriter.add_scalars('testing/prediction losses',{'E prediction loss':EPredictionLoss/count,'SE prediction loss':SEPredictionLoss/count},epoch)
            tensorboardWriter.flush()

            print()
            print('='*20,end = ' ')
            print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
            using_time = time.time()-start_time
            hours = int(using_time/3600)
            using_time -= hours*3600
            minutes = int(using_time/60)
            using_time -= minutes*60
            print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)),end = ' ')
            print('='*20)
            print()


    if TestOrTrain == 'test':

        modelParamFolder = '/home/hsc/Research/StateMapPrediction/code/models/mirrorAE/resultDir/20191206_16_02_40/modelParam'
        typicalTestDataset = typicalTestData(E_dataset_path,SE_dataset_path)
        typicalTestDataLoader = DataLoader(typicalTestDataset,batch_size=4,shuffle=False)

        EastModel = BehaviorModelAutoEncoder()
        SouthEastModel = BehaviorModelAutoEncoder()

        EastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'Easemodel.pth')))
        SouthEastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'SEmodel.pth')))
        EastModel.to(device)
        SouthEastModel.to(device)
        criterion = nn.MSELoss()

        
        zzlossedAVG = 0
        for fuck in range(20):
            zzlossed = []
            coefficent = []
            for testDeltaT in range(9):

                fakeSingleTrainset = FakeDeltaTDataset(E_dataset_path,SE_dataset_path,testDeltaT,train = True)
                typicalTestDataLoader = DataLoader(fakeSingleTrainset,batch_size=4,shuffle=False)
                count = 0
                totalzzloss = 0
                print('processing deltaT = %d'%(testDeltaT))

                for i,sample in enumerate(typicalTestDataLoader):
                    E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)

                    Ez = EastModel.encoder(E)
                    EinSout = SouthEastModel.decoder(Ez)
                    EinEout = EastModel.decoder(Ez)

                    Sz = SouthEastModel.encoder(SE)
                    SinEout = EastModel.decoder(Sz)
                    SinSout = SouthEastModel.decoder(Sz)

                    # concatenate = torch.cat([E,SE,EinEout,SinSout,SinEout,EinSout],0)
                    # concatenate = torch.cat([E,SE,SinEout,EinSout],0)
                    # concatenate = concatenate.detach()
                    # concatenate = concatenate.cpu()
                    # concatenate = convertDataToBGR(concatenate)
                    # concatenate = torchvision.utils.make_grid(concatenate,nrow=8,normalize=False,pad_value=255)

                    # concatenate = concatenate.numpy()
                    # concatenate = np.transpose(concatenate,(1,2,0))
                    # imgName = '/home/hsc/typicalTestResult.jpg'
                    # cv2.imwrite(imgName,concatenate)
                    # print('write img to ', imgName)

                    loss = criterion(Ez,Sz)


                    Ez = Ez.detach()
                    Ez = Ez.cpu()
                    Ez = Ez.numpy()
                    Sz = Sz.detach()
                    Sz = Sz.cpu()
                    Sz = Sz.numpy()

                    totalzzloss += loss.item()
                    count += 1
                    # print('z-z mse = ',loss.item())

                # print('testDeltaT = %d,z-z mse = %f '%(testDeltaT,totalzzloss/count))
                zzlossed.append(totalzzloss/count)
                coefficent.append(np.exp(testDeltaT*0.55))
            
            zzlossed = np.array(zzlossed)
            print(zzlossed)
            print()
            # zzlossed = zzlossed/zzlossed[0]
            zzlossedAVG += zzlossed
        zzlossedAVG/=20

        print(zzlossedAVG/zzlossedAVG[0])
        # print(coefficent)    


            

        

        
