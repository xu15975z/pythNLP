import pandas as pd

import matplotlib.pyplot as plt

G_loss_list = pd.read_csv("Load_model/G_loss_list.csv")
#L_loss_list = pd.read_csv("Load_model/L_loss_list.csv")
#plt.plot(L_loss_list, label="BiLSTM")
plt.plot(G_loss_list, label="BiGRU")
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.legend()
plt.show()