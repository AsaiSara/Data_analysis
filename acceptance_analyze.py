
# coding: utf-8

# # 受諾度ラベルの割合を確認
# * 受諾度ごとに発話文を表示(only user utterance)
# * 石川さんの論文には書いていない

# In[2]:


from collections import Counter
import pickle
import sentencepiece as spm
from matplotlib import pyplot as plt
import numpy as np
import re

def sentence_piece(vocab_path):
    sp = spm.SentencePieceProcessor()
    sp.Load(vocab_path)
    n_words = len(sp)
    return sp, n_words


# In[3]:


id2emo = {"0":"NONE","1":"neutral","2":"anger","3":"sad","4":"happy","5":"contentment"}


# In[250]:


emo_list = [id2emo[str(i)] for i in range(1,6)]
acc_label = np.arange(1,6)
print("emotion list : ",emo_list)
print("accuracy list : ", acc_label)


# In[247]:


def read_acc_dialog(from_path):
    file_dial = open(from_path+"_dial.txt", "r").readlines()
    file_acc = open(from_path+"_accept.txt", "r").readlines()
    file_emo = open(from_path+"_emotion.txt", "r").readlines()
    dialog = [""]*5
    count_acc = np.array([[0]*5]*5) #acceptance_size:5, emotional_size:5, 
    user_utter = [utter_pair.split("\t")[0] for utter_pair in file_dial]
    user_acc = [int(acc_pair.split("\t")[0]) for acc_pair in file_acc]
    user_emo = [int(emo_pair.split("\t")[0]) for emo_pair in file_emo]
    system_emo = [int(emo_pair.split("\t")[1]) for emo_pair in file_emo]

    for i in range(len(user_acc)): # i is the number of utterance
        if system_emo[i] != 0 and user_emo[i] != 0:
            dialog[user_acc[i]-1] += user_utter[i] + "\n" # acc was from 1 to 5 -> 0 to 4
            count_acc[user_acc[i]-1][user_emo[i]-1] += 1
    print(count_acc)
    for i in range(count_acc.shape[1]):
        print("\t\temotion : {} ({} utterances)".format(i+1, sum([count_acc[j][i] for j in range(5)])))
    for i in range(count_acc.shape[0]):
        print("\t\tacceptance : {} ({} utterances)".format(id2emo[str(i+1)], sum(count_acc[i])))             
    return count_acc, user_acc, system_emo


# * Path of dialogues

# In[264]:


from_dir = "../data/em_dial/splitted/"
domain = ["/cleaning","/exercise","/lunch","/sleep","/game","/all"]
last_part = ["/train","/valid","/test"]

acc_per_domain = []
user_acc = []
system_emo = []

for i,l in enumerate(last_part):
    print("\ttype of data : "+l)  
    acc_sum = []
    for j,d in enumerate(domain[:-1]):
        print("Domain : "+d) 
        acc_per_domain.append(read_acc_dialog(from_dir+d+l)[0])
        user_acc+=read_acc_dialog(from_dir+d+l)[1]
        system_emo+=read_acc_dialog(from_dir+d+l)[2]
        acc_sum.append([sum(acc) for acc in acc_per_domain[-1]])  
    #dl = [d+l for d in domain for i in last_part]
    print(acc_sum)
    #print(acc_sum[i*len(domain[:-1]):])
    graph_plot(acc_label, acc_sum, domain)
    #graph_plot(np.arange(1,6), acc_sum[i*len(domain[:-1]):])
    plt.show()


# In[241]:


#for i wr in range enumerate(wr1):
import numpy as np

def graph_plot(labels, acc_per_emo, label, max_y=None):
    print(acc_per_emo)
    labels = labels
    width = 0.15
    left = np.arange(len(acc_per_emo[0]))
    #x = np.arange(0, len(acc_per_emo), 1)
    #y = np.array(acc_per_emo)

    plt.title("acceptant level")
    plt.xlabel("acceptance rate")
    plt.ylabel("frequency of appearance")
 
    for i in range(len(labels)):
        print(acc_per_emo[i])
        plt.bar(left+width*i, acc_per_emo[i], width=width, align='center',label=label[i])
        plt.legend(loc='best',shadow=True)
    #plt.ylim(0,max_y)
    plt.xticks(left+width*2, labels)
    #plt.show


# In[242]:


for j,d in enumerate(domain[:-1]):
    print("Domain : "+d)    
    for i,l in enumerate(last_part):   
        print("\ttype of data : "+l)
        graph_plot(emo_list, acc_per_domain[i+j*i], acc_label)
        plt.show()


# In[261]:


emo_acc = np.array([[0]*5]*5)
for i in range(len(system_emo)-1)
    s = system_emo[i]
    u = user_acc[i+1]
        for u in user_acc[1:]:
            emo_acc[s-1][u-1]+=1
        emo_acc[]


# In[266]:


Counter(system_emo)


# In[255]:


for i in range(len(user_acc)-1):
    print(acc_per_emo[i])
    plt.bar(left+width*i, acc_per_emo[i], width=width, align='center',label=label[i])
    plt.legend(loc='best',shadow=True)

