
from Weibo_GRULSTM.Weibo_dataset_3 import fenci

#创造字典
class wordSequence:

    UNK_T="UNK"#字典里没有的词 用UNK作为键
    PAD_T="PAD"#作为参数的句子如果长度不狗用PAD作为键然后填充
    UNK=0
    PAD=1
    #0，1 分别是他们的索引
    def __init__(self):
        self.dic={#将词转换成数字的字典  键为词作为索引  值为数
            self.UNK_T:self.UNK,
            self.PAD_T:self.PAD
            #为字典赋初值
        }
        self.count={}#统计词频的集合

    def __len__(self):
        return len(self.dic)

    def fit(self,sentence):
        for word in sentence:
            self.count[word]= self.count.get(word,0)+1#get（word，0） 获取集合中建为word的值 没有就取0
            #遍历sentence列表中的每个单词  如果在统计词频的集合已经存在就+1
        #print(self.count)

    def build_voc(self, min=1, max=None , max_features=None):#创建真正的词典
        """
        :param min: 如果没达到最小词频 就不出现在字典
        :param max: 最大词频
        :param max_features: 字典最大容量
        :return:

        """
        # 遍历词频集合  删选不符合要求的单词
        # val是词频
        if min is not None:
            self.count = {word: val for word,val in self.count.items() if val>min}
        if max is not None:
            self.count = {word: val for word, val in self.count.items() if val < max}
        # ↓按词频排序重新放入词频字典
        if max_features is not None:
            self.count = dict(sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features])

        # sorted（）函数举例:
        # >>> students = [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
        # >>> sorted(students, key=lambda s: s[2])            # 按年龄排序
        # [('dave', 'B', 10), ('jane', 'B', 12), ('john', 'A', 15)]
        #
        # >>> sorted(students, key=lambda s: s[2], reverse=True)       # 按降序
        # [('john', 'A', 15), ('jane', 'B', 12), ('dave', 'B', 10)]
        for word in self.count:
            self.dic[word]=len(self.dic)#放入字典
        #反向字典↓
        self.inverse_dic = dict(zip(self.dic.values(),self.dic.keys()))

    def tranfroms(self,sentences,maxlen=None):#将句子转换成数
        l = len(sentences)
        if maxlen is not None:
            if maxlen > l:
                sentences = sentences+[self.PAD_T]*(maxlen-l)#长度不够填空白
            if maxlen < l:
                sentences = sentences[:maxlen]
        return [self.dic.get(word,self.UNK) for word in sentences]

    def In_tranfroms(self,indices):#数转句子
        return [self.inverse_dic.get(idx) for  idx in indices]


# if __name__=='__main__':
#       #path=r"F:\pythonProject\NLP\aclImdb\test\neg\0_2.txt"
#       list=fenci("我是中国人我是中国人我是中国人我是中国人我是中国人我是中国人我是中国人")
#       #print(list)
#       ws=wordSequence()
#       ws.fit(list)
#       ws.build_voc()
#       res=ws.tranfroms(list)
#       print(res)
#       #print(ws.In_tranfroms(res))
#       print(ws.dic)

