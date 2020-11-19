import torch
import numpy as np
from Weibo_GRULSTM.Weibo_dataset_3 import get_Dl
import torch.nn.functional as F
from Weibo_GRULSTM.load_dict import ws,maxlen
import os
from tqdm import tqdm

from sklearn import metrics
import matplotlib.pyplot as plt
from Weibo_GRULSTM.Config import *



class GRUModel(torch.nn.Module):
    def __init__(self):
        super(GRUModel,self).__init__()

        self.embedding = torch.nn.Embedding(len(ws),embedd_dim)
        self.laynorm=torch.nn.LayerNorm([batch_size,maxlen,embedd_dim])
        self.GRU= torch.nn.GRU(num_layers=num_layer,hidden_size=hidden_size,
                                input_size=100, batch_first=True, bidirectional=bidriection,
                                dropout=0.5)

        self.fc = torch.nn.Linear(hidden_size*2,2)

    def forward (self,input):
        # print(111)
        x = self.embedding(input)
        # batch_size，max_len，100
        # torch.Size([128, 150, 100])
        x = self.laynorm(x)
        x, h_n =self.GRU(x)
        # x:batch_size*max_len*  2*hid_len(双向）
        # h_n:2*2(两层双向) * batch_size*hidden_size
        # output_forwd = h_n[:,-2,:]# 正向最后一次输出
        # output_bacwd = h_n[:,-1,:]# 反向最后一次输出
        output=torch.cat([h_n[-1], h_n[-2]], dim=1)# batch_size *hidden_size*2

        out=self.fc(output)
        #c=F.log_softmax(out,dim=-1)
        return F.log_softmax(out,dim=-1)