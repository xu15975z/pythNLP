import os
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from Weibo_GRULSTM.Weibo_dataset_3 import get_Dl
from Weibo_GRULSTM.GRU_model_5 import GRUModel
from Weibo_GRULSTM.LSTM_model_4 import LSTMModel
import pandas as pd

Gmodel = GRUModel()
Lmodel = LSTMModel()
Goptimizer = torch.optim.Adam(Gmodel.parameters(), 0.001)
Loptimizer = torch.optim.Adam(Lmodel.parameters(), 0.001)

if os.path.exists("/Load_model/GRUmodule.pkl"):
      Gmodel.load_state_dict(torch.load("../Weibo_GRULSTM/Load_model/GRUmodule.pkl"))
      Goptimizer.load_state_dict(torch.load("../Weibo_GRULSTM/Load_model/GRUoptimizer.pkl"))
#
# if os.path.exists("../Weibo_GRULSTM/Load_model/LSTMmodule.pkl"):
#      Lmodel.load_state_dict(torch.load("../Weibo_GRULSTM/Load_model/LSTMmodule.pkl"))
#      Loptimizer.load_state_dict(torch.load("../Weibo_GRULSTM/Load_model/LSTMoptimizer.pkl"))

def get_time_dif (start_time) :
    end = time.time()
    time_dif = end - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))

def Gtrain(epoh,dataloader) :

    G_loss_list = []
    data_loader = dataloader

    start_time = time.time()
    Gmodel.train()
    #for epoch
    for idx, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        Goptimizer.zero_grad()
        #input -= np.mean(input, axis=0)  # zero-center
        #input =torch.nn.
        out = Gmodel(input)
        loss = F.nll_loss(out, target)
        G_loss_list.append(loss.item()*10)
        loss.backward()
        Goptimizer.step()

        if idx % 10 == 0:
            print(epoh, idx, loss.item())

        if idx % 150 == 0 and idx != 0:
            torch.save(Gmodel.state_dict(), "../Weibo_GRULSTM/Load_model/GRUmodule.pkl")
            torch.save(Goptimizer.state_dict(), "../Weibo_GRULSTM/Load_model/GRUoptimizer.pkl")
    G_time_dif = get_time_dif(start_time)
    print("GRU训练完成")
    return G_time_dif,G_loss_list


def Ltrain(epoh,dataloader):
    L_loss_list = []
    data_loader = dataloader
    start_time = time.time()
    for idx, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
        Loptimizer.zero_grad()
        out = Lmodel(input)
        loss = F.nll_loss(out, target)
        L_loss_list.append(loss.item() * 10)
        loss.backward()
        Loptimizer.step()

        if idx % 75 == 0:
            print(epoh, idx, loss.item())

        if idx % 150 == 0 and idx != 0:
            torch.save(Lmodel.state_dict(), "../Weibo_GRULSTM/Load_model/LSTMmodule.pkl")
            torch.save(Loptimizer.state_dict(), "../Weibo_GRULSTM/Load_model/LSTMoptimizer.pkl")
    L_time_dif = get_time_dif(start_time)
    return L_time_dif, L_loss_list

def eval():
    target_list=["0","1"]
    predict_all = np.array([], dtype=int)
    lables_all = np.array([], dtype=int)
    data_loader =get_Dl(train=False)
    Gmodel.eval()
    for idx,(input,target) in tqdm(enumerate(data_loader),total=len(data_loader)):
        with torch.no_grad():
            output=Gmodel(input)
            #准确率
            pred=output.max(dim=-1)[-1]
            target = target.data.numpy()
            lables_all = np.append(lables_all, target)
            pred=pred.numpy()
            predict_all = np.append(predict_all, pred)
    acc=metrics.accuracy_score(lables_all, predict_all)

    report = metrics.classification_report(lables_all, predict_all, target_names=target_list, digits=4)
    confusion = metrics.confusion_matrix(lables_all, predict_all)
    print("准确率")
    print(acc)
    print("混淆矩阵：")
    print(confusion)
    print("测试报告和F1")
    print(report)

if __name__=='__main__':
   # GRU与LSTM的对比试验
    np.random.seed(1)
    torch.manual_seed(1)
    dataload=get_Dl(train=True)
    #
    G_time_dif, G_loss_list = Gtrain(1, dataload)
    #L_time_dif, L_loss_list = Ltrain(1, dataload)

    # df_L = pd.DataFrame(L_loss_list)
    # df_L.to_csv('./Load_model/L_loss_list.csv', mode='a', header=False, index=None)
    df_G = pd.DataFrame(G_loss_list)
    df_G.to_csv('./Load_model/G_loss_list.csv', mode='a', header=False, index=None)

    # print("LSTM所用时间", L_time_dif, "GRU所用时间", G_time_dif)
    # print("GRU所用时间为LSTM的", (G_time_dif / L_time_dif) * 100, "%")
    # plt.plot(L_loss_list, label="BiLSTM")
    # plt.plot(G_loss_list, label="BiGRU")
    # plt.ylabel('Loss')
    # plt.xlabel('Iteration')
    # plt.legend()
    # plt.show()

    eval()

