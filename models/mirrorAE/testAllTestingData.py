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


modelParamFolder = '/home/hsc/Research/StateMapPrediction/code/models/mirrorAE/resultDir/datasetChanged/20200107_11_38_41/modelParam'#网络参数路径
savePath = '/home/hsc/Research/StateMapPrediction/code/models/mirrorAE/resultDir/predictMinusGT/AE/0%0.5'#把图片保存的路径

E_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data7'#数据集路径
SE_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data7'

if not os.path.exists(savePath):
    os.makedirs(savePath)



# 加载模型
EastModel = BehaviorModelAutoEncoder(2,0)#这里的参数可能需要更改，注意一下
SouthEastModel = BehaviorModelAutoEncoder(2,0)
EastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'Easemodel.pth')))
SouthEastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'SEmodel.pth')))
EastModel.eval()
SouthEastModel.eval()
device = torch.device('cuda:0')
EastModel.to(device)
SouthEastModel.to(device)

fakeSingleTestset = FakeDeltaTDataset(E_dataset_path,SE_dataset_path,0,4, train = False)
fakeSingleTestLoader = DataLoader(fakeSingleTestset,batch_size=24,shuffle=False) #测试集有72张，总共需要保存3张大图

for i,sample in enumerate(fakeSingleTestLoader):
    E,SE = sample['EStateMap'].to(device), sample['SEStateMap'].to(device)
    Ez = EastModel.encoder(E)
    EinSout = SouthEastModel.decoder(Ez)

    Sz = SouthEastModel.encoder(SE)
    SinEout = EastModel.decoder(Sz)

    SE = SE.detach()
    SE = SE.cpu()
    E = E.detach()
    E = E.cpu()

    # 保存东南门的预测结果
    concatenate = EinSout
    concatenate = concatenate.detach()
    concatenate = concatenate.cpu()
    minus = concatenate - SE#预测值与真值做差
    concatenate = convertDataToBGR(concatenate)
    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=False,pad_value=0)

    # 保存预测结果可视化图
    concatenate = concatenate.numpy()
    concatenate = np.transpose(concatenate,(1,2,0))
    imgName = os.path.join(savePath,'SE%d.png'%i)
    cv2.imwrite(imgName,concatenate)
    print('write img to ', imgName)

    # 将做差后的结果可视化保存
    minus = minus.numpy()
    minus = np.abs(minus)
    minus = np.sum(minus,axis = 1)
    minus = minus[:,np.newaxis,:,:]
    minus = minus*2
    minus[minus>255] = 255
    minus = 255-minus
    minus = torch.Tensor(minus)
    minus = torchvision.utils.make_grid(minus,nrow = 4, normalize=False,pad_value=2)
    minus = minus.numpy()
    minus = np.transpose(minus,(1,2,0))
    imgName = os.path.join(savePath,'SEminus%d.png'%i)
    cv2.imwrite(imgName,minus)
    print('write img to ', imgName)
    

    # 保存东门的预测结果
    concatenate = SinEout
    concatenate = concatenate.detach()
    concatenate = concatenate.cpu()
    minus = concatenate - E#预测值与真值做差
    concatenate = convertDataToBGR(concatenate)
    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=False,pad_value=0)

    concatenate = concatenate.numpy()
    concatenate = np.transpose(concatenate,(1,2,0))
    imgName = os.path.join(savePath,'E%d.png'%i)
    cv2.imwrite(imgName,concatenate)
    print('write img to ', imgName)

    # 将做差后的结果可视化保存
    minus = minus.numpy()
    minus = np.abs(minus)
    minus = np.sum(minus,axis = 1)
    minus = minus[:,np.newaxis,:,:]
    minus = minus*2
    minus[minus>255] = 255
    minus = 255-minus
    minus = torch.Tensor(minus)
    minus = torchvision.utils.make_grid(minus,nrow = 4, normalize=False,pad_value=2)
    minus = minus.numpy()
    minus = np.transpose(minus,(1,2,0))
    imgName = os.path.join(savePath,'Eminus%d.png'%i)
    cv2.imwrite(imgName,minus)
    print('write img to ', imgName)