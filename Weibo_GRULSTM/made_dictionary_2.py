#构造字典
from tqdm import tqdm
from Weibo_GRULSTM.Weibo_dataset_3 import fenci
import pickle
from Weibo_GRULSTM.creat_dictionary_1 import wordSequence
import pandas as pd
import numpy as np

if __name__== '__main__':
    ws = wordSequence()
    path="data/train.csv"
    df=pd.read_csv(path)
    input_list = df["data"]
    #lable_list = df["label"]


    for input in input_list:
        sentence = fenci(input)
        ws.fit(sentence)
    ws.build_voc(min=10)

    pickle.dump(ws, open("/data/Dictionary.pkl", "wb"))
    #print(ws.dic)