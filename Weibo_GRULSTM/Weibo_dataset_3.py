from torch.utils.data import Dataset, DataLoader
import torch
import jieba
import os
from sklearn.model_selection import train_test_split
import pickle
import re
import pandas as pd


def fenci(content):  # 分词函数
    res = jieba.lcut(content)
    # strip()去除空格  lower()变小写
    return res


class Weibo_data(Dataset):  # Dataset类
    def __init__(self, train=True):
        # 读进来全部原文 做成list

        if train:
            df = pd.read_csv("data/train.csv")
        else:
            df = pd.read_csv("data/test.csv")

        self.data_list = df["data"]
        self.label_list = df["label"]
        #print("weibo___dataset")
        #self.ws=pickle.load(open("Dictionary.pkl","rb"))
        #print(1)

    def __getitem__(self, index):
        # 将读取出的这一条分词
        content = self.data_list[index]
        content = fenci(content)
        labe=self.label_list[index]


        return content, labe  # 返回内容和对应的标签
        # print(lable)

    def __len__(self):

        return len(self.label_list)


def collate_fn(batch):  # 重写dataloader里的方法
    content, label = list(zip(*batch))
    # list(zip(*batch))
    from Weibo_GRULSTM.load_dict import ws, maxlen  # 读取之前保存的词典
    content = [ws.tranfroms(i, maxlen=maxlen) for i in content]  # 转换
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)

    return content, label


def get_Dl(train=True):  # 获取dataloader
    weibo_data = Weibo_data(train)

    data_loder = DataLoader(dataset=weibo_data,
                            batch_size=128, shuffle=True, drop_last=True,
                            collate_fn=collate_fn)
    return data_loder


if __name__ == '__main__':
    # path=r"F:\pythonProject\NLP\aclImdb\train\neg\12391_4.txt"
    # print(fenci(open(path).read()))

    for idx, (input, target) in enumerate(get_Dl()):
        print(idx)
        print(input)
        print(target)
        break

