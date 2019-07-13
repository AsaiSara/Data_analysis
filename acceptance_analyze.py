
# coding: utf-8

# # 受諾度ラベルの割合を確認
# * 受諾度ごとに発話文を表示(only user utterance)
# * 石川さんの論文には書いていない

# In[395]:


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


# In[411]:


import matplotlib
matplotlib.colors.cnames["limegreen"]


# In[485]:


import matplotlib
matplotlib.colors.cnames["grey"]


# In[486]:


id2emo = {"0":"NONE","1":"neutral","2":"anger","3":"sad","4":"happy","5":"contentment"}


# In[487]:


emo_list = [id2emo[str(i)] for i in range(1,6)]
emo_color = ["#808080","#e41a1c","#377eb8","#ff7f00","#4daf4a"]

acc_label = np.arange(1,6)
print("emotion list : ",emo_list)
print("accuracy list : ", acc_label)


# In[522]:


def read_acc_dialog(from_path, print_num=None):
    file_dial = open(from_path+"_dial.txt", "r").readlines()
    file_acc = open(from_path+"_accept.txt", "r").readlines()
    file_emo = open(from_path+"_emotion.txt", "r").readlines()
    dialog = [""]*5
    dialog_us = [""]*5
    user_emo = []
    sys_emo = []
    
    count_acc = np.array([[0]*5]*5) #acceptance_size:5, emotional_size:5, 
    user_utter = [utter_pair.split("\t")[0] for utter_pair in file_dial]
    user_acc = [int(acc_pair.split("\t")[0]) for acc_pair in file_acc]
    user_emo = [int(emo_pair.split("\t")[0]) for emo_pair in file_emo]
    system_emo = [int(emo_pair.split("\t")[1]) for emo_pair in file_emo]
    u_s_utter = [utter_pair for utter_pair in file_dial]
    delet_uttr = 0

    if len(user_utter) != len(u_s_utter):
        return None
    
    for i in range(len(user_acc)): # i is the number of utterance
        #print(len(user_acc))
        if system_emo[i] != 0:
            delet_uttr += 1
            dialog[user_acc[i]-1] += user_utter[i] + "\n" # acc was from 1 to 5 -> 0 to 4
            dialog_us[user_acc[i]-1] += str(system_emo[i]) + "\t" + u_s_utter[i]  
            count_acc[user_acc[i]-1][user_emo[i]-1] += 1
        #else:
         #   system_emo.pop(i) 
          #  user_emo.pop(i) 
    if print_num==True:
        for i in range(count_acc.shape[1]):
            print("\t\tacceptance : {} ({} utterances)".format(i+1, sum([count_acc[j][i] for j in range(5)])))
        for i in range(count_acc.shape[0]):
            print("\t\temotion : {} ({} utterances)".format(id2emo[str(i+1)], sum(count_acc[i]))) 
        print("\tnumber of all utterance : ", len(user_acc))
        print("\texcept of NONE label : ", delet_uttr)
    return count_acc, user_acc, system_emo, user_utter, system_utter, dialog, dialog_us


# * Path of dialogues

# In[523]:


#for i wr in range enumerate(wr1):
import numpy as np

def graph_plot(X_labels, acc_per_emo, label, max_y=None,
               x_label="Acceptance label", y_label="Number of utterancce",title_name="Corpus analysis"):
    #print(acc_per_emo)
    labels = X_labels
    width = 0.15
    left = np.arange(len(acc_per_emo[0]))
    #x = np.arange(0, len(acc_per_emo), 1)
    #y = np.array(acc_per_emo)

    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
 
    for i in range(len(labels)):
        #print(acc_per_emo[i])
        plt.bar(left+width*i, acc_per_emo[i], width=width, align='center',label=label[i])
        plt.legend(loc='best',shadow=True)
    #plt.ylim(0,max_y)
    plt.xticks(left+width*2, labels)
    #plt.show


# ## train/valid/testで分けた場合

# In[524]:


from_dir = "../data/em_dial/splitted/"
domain = ["/cleaning","/exercise","/lunch","/sleep","/game","/all"]
last_part = ["/train","/valid","/test"]

acc_per_domain = []
user_acc = []
system_emo = []
user_utter = []
system_utter = []
dialog_per_acc = []
dialog_per_acc_us = []
for i,l in enumerate(last_part):
    print("\ttype of data : "+l)  
    acc_sum = []
    for j,d in enumerate(domain[:-1]):
        print("Domain : "+d) 
        acc_per_domain.append(read_acc_dialog(from_dir+d+l)[0])
        user_acc.append(read_acc_dialog(from_dir+d+l)[1])
        system_emo.append(read_acc_dialog(from_dir+d+l)[2])
        user_utter.append(read_acc_dialog(from_dir+d+l)[3])
        system_utter.append(read_acc_dialog(from_dir+d+l)[4])
        dialog_per_acc.append(read_acc_dialog(from_dir+d+l)[5])
        dialog_per_acc_us.append(read_acc_dialog(from_dir+d+l,print_num=True)[6])
        acc_sum.append([sum(acc) for acc in acc_per_domain[-1]])  # 今追加した要素だけ使うので -1
    #dl = [d+l for d in domain for i in last_part]
    #print(acc_sum[i*len(domain[:-1]):])
    graph_plot(acc_label, acc_sum, domain)
    #graph_plot(np.arange(1,6), acc_sum[i*len(domain[:-1]):])
    plt.show()


# ## データセットを分けない場合

# In[525]:


def concat_last_part(data, last_len=3, domain_len=5):
    # last_len * domain_len, ???
    data2 = [0]*domain_len
    for d in range(domain_len):
        data2[d] = data[d]+data[d+domain_len]+data[d+domain_len*2]
    return data2


# In[526]:


acc_per_domain2 = concat_last_part(acc_per_domain)
user_acc2 = concat_last_part(user_acc)
system_emo2 = concat_last_part(system_emo)
user_utter2 = concat_last_part(user_utter)
system_utter2 = concat_last_part(system_utter)
acc_sum2=[]
for d in range(len(domain[:-1])):
    acc_sum2.append([sum(acc) for acc in acc_per_domain2[d]])
graph_plot(acc_label, acc_sum2, domain)


# In[493]:


print(system_emo[0][:10])


# ## 受諾度の4と5は一緒にしてしまうのが良さそう

# ### usはuser, system両方の発話のこと

# In[494]:


dialog2 = []
dialog2_us = []
domain_len = len(domain[:-1])
for d in range(domain_len):
    print("Domain : ", domain[d])
    dial = [""]*len(acc_label) 
    dial_us = [""]*len(acc_label)
    for a in range(len(acc_label)):
        per_acc = dialog_per_acc[d][a]+dialog_per_acc[d+domain_len][a]+dialog_per_acc[d+domain_len*2][a]
        per_acc_us = dialog_per_acc_us[d][a]+dialog_per_acc_us[d+domain_len][a]+dialog_per_acc_us[d+domain_len*2][a]
        dial[a] += per_acc
        dial_us[a] +=  per_acc_us
        
        dial_us_split = [p.split("\t") for p in per_acc_us.split("\n") if p != ""]
        count_utter = Counter(per_acc.split("\n")).most_common()

        print("\tacceptance : ", a+1)#, len(count_utter), len(dial_us_split))
        for i in range(5):
            print("\n\t",count_utter[i][0]," ",count_utter[i][1], "pairs ", )
            for us in dial_us_split:
                #if len(us) != 2:
                #print(us)
                if count_utter[i][0] == us[1]:
                    print("  ユーザ「{}」\nシステム「{}」({})".format(us[1], us[2], id2emo[us[0]]))
                    print("_____________________________________________________________________")
                    #print(" \t\t", us, " ", id2emo[str(system_emo[d][i])])
    dialog2.append(dial)
    dialog2_us.append(dial_us)


# # 受諾度が高いときの前の感情が気になる

# In[530]:


def read_acc_dialog_3pairs(from_path, print_num=None):
    file_dial = open(from_path+"_dial.txt", "r").readlines()
    file_acc = open(from_path+"_accept.txt", "r").readlines()
    file_emo = open(from_path+"_emotion.txt", "r").readlines()
    dialog = [""]*5
    dialog_us = [""]*5
    user_emo = []
    sys_emo = []
    delet_uttr = 0
    
    #print(len(file_emo))
    
    count_acc = np.array([[0]*5]*5) #acceptance_size:5, emotional_size:5, 
    user_utter = [utter_pair.split("\t")[1] for utter_pair in file_dial]
    user_acc = [int(acc_pair.split("\t")[1]) for acc_pair in file_acc]
    user_emo = [int(emo_pair.split("\t")[1]) for emo_pair in file_emo]
    system_emo1 = [int(emo_pair.split("\t")[0]) for emo_pair in file_emo]
    system_emo2 = [int(emo_pair.split("\t")[2]) for emo_pair in file_emo]
    u_s_utter = [utter_pair for utter_pair in file_dial]

    if len(user_utter) != len(u_s_utter):
        return None
    
    for i in range(len(user_acc)): # i is the number of utterance
        if system_emo1[i] != 0 and system_emo2[i] != 0 and user_emo[i] != 0:
            dialog[user_acc[i]-1] += user_utter[i] + "\n" # acc was from 1 to 5 -> 0 to 4
            dialog_us[user_acc[i]-1] += str(system_emo1[i]) + "\t" + u_s_utter[i] + "\t" + str(system_emo2[i]) 
            count_acc[user_acc[i]-1][user_emo[i]-1] += 1
            delet_uttr +=1
        #else:
         #   system_emo.pop(i) 
          #  user_emo.pop(i) 
    #print(count_acc)
    if print_num==True:
        for i in range(count_acc.shape[1]):
            print("\t\temotion : {} ({} utterances)".format(i+1, sum([count_acc[j][i] for j in range(5)])))
        for i in range(count_acc.shape[0]):
            print("\t\tacceptance : {} ({} utterances)".format(id2emo[str(i+1)], sum(count_acc[i])))  
        print("\tnumber of all utterance : ", len(user_acc))
        print("\texcept of NONE label : ", delet_uttr)
    return count_acc, user_acc, system_emo1, system_emo2, user_utter, system_utter, dialog, dialog_us


# In[531]:


from_dir_3pair = "../data/em_dial/splitted/sys_user_sys"
domain_3pair = ["/cleaning","/exercise","/lunch","/sleep","/game","/all"]
last_part_3pair = ["/all","/train","/valid","/test"]

acc_per_domain3 = []
user_acc3 = []
system_emo3_1 = []
system_emo3_2 = []
user_utter3 = []
system_utter3 = []
dialog_per_acc3 = []
dialog_per_acc_us3 = []
for i,l in enumerate(last_part_3pair):
    print("\ttype of data : "+l)  
    acc_sum3 = []
    for j,d in enumerate(domain_3pair[:-1]):
        print("Domain : "+d) 
        acc_per_domain3.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[0])
        user_acc3.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[1])
        system_emo3_1.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[2])
        system_emo3_2.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[3])
        user_utter3.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[4])
        system_utter3.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[5])
        dialog_per_acc3.append(read_acc_dialog_3pairs(from_dir_3pair+d+l)[6])
        dialog_per_acc_us3.append(read_acc_dialog_3pairs(from_dir_3pair+d+l,print_num=True)[7])
        acc_sum3.append([sum(acc) for acc in acc_per_domain3[-1]])  # 今追加した要素だけ使うので -1
    #dl = [d+l for d in domain for i in last_part
    #print(acc_sum[i*len(domain[:-1]):])
    graph_plot(acc_label, acc_sum3, domain)
    #graph_plot(np.arange(1,6), acc_sum[i*len(domain[:-1]):])
    plt.show()


# In[535]:


#for i wr in range enumerate(wr1):
import numpy as np

def graph_plot(X_labels, acc_per_emo, label, max_y=None, color_list=emo_color,
               x_label="Acceptance label", y_label="Number of utterancce",title_name="Corpus analysis"):
    labels = X_labels
    width = 0.15
    left = np.arange(len(acc_per_emo[0]))

    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
 
    for i in range(len(labels)):
        #print(collor_list*5)
        plt.bar(left+width*i, acc_per_emo[i], color=color_list[i], width=width, align='center',label=label[i])
        plt.legend(loc='best',shadow=True)
    #plt.ylim(0,max_y)
    plt.xticks(left+width*2, labels)
    #plt.show


# In[536]:


def former_emotion(user_acc, system_emo, Plot_type_2 = False, Plot2_ratio = False, max_y=None, color_list=emo_color,
                   x_label="Acceptance label", y_label="Number of utterancce",title_name="Corpus analysis"):
    acc_emo_list = np.array([[0]*5]*5) # acception, emotion ... 5 * 5 
    t = 0
    for i in range(len(user_acc)):
        if system_emo[i] != 0:
            acc_emo_list[user_acc[i]-1][system_emo[i]-1] += 1
        else:
            t += 1
    print("\tNone label : ",t)
    
    if Plot_type_2 ==True:
        graph_plot2(acc_label, acc_emo_list, emo_list,
                    ratio = Plot2_ratio, max_y = max_y, color_list=color_list,
                    x_label=x_label, y_label=y_label, title_name=title_name)
    else:
        graph_plot(acc_label, acc_emo_list, emo_list, color_list=color_list,
                   x_label=x_label, y_label=y_label, title_name=title_name)
    plt.show()


# In[537]:


for j in range(len(domain[:-1])):
    print("Domain : "+domain[j])
    former_emotion(user_acc3[j], system_emo3_1[j])


# ## ドメインを分けない場合

# In[459]:


def concat_last_part(data, last_len=3, domain_len=5):
    # last_len * domain_len, ???
    data2 = []*len(emo_list)
    for e in range(len(emo_list)):
        data2[e] = data[d]+data[d+domain_len]+data[d+domain_len*2]
    return data2


# In[538]:


user_acc3_all = []
system_emo3_1_all = []
system_emo3_2_all = []
for i in range(5):
    user_acc3_all += user_acc3[i]
    system_emo3_1_all += system_emo3_1[i]
    system_emo3_2_all += system_emo3_2[i]
print("Former system emotion : ")
former_emotion(user_acc3_all, system_emo3_1_all, y_label="Number of emotion labels")
print("Later system emotion : ")
former_emotion(user_acc3_all, system_emo3_2_all, y_label="Number of emotion labels")


# In[502]:


def graph_plot2(X_labels, acc_per_emo, label, max_y=None, ratio=False, color_list=emo_color,
               x_label="Acceptance label", y_label="Number of utterancce",title_name="Corpus analysis"):
    
    labels = X_labels
    left = np.arange(len(acc_per_emo[0]))
    plt.title(title_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    bottom= [0]*len(labels)
    if ratio == True:
        former = acc_per_emo  # shape of acc_per_emo:  label(emotion) size * X_labels(acceptance) size
         # -> X_labels size ... sum of label(emotion) size per acceptance
        sum_acc = [sum([acc_per_emo[acc][emo] for emo in range(len(label))]) for acc in range(len(X_labels))]
        acc_per_emo = [[acc_per_emo[acc][emo]/sum_acc[acc] for emo in range(len(label))]
                       for acc in range(len(X_labels))]
        print(sum(acc_per_emo[0]))
        for i, acc in enumerate(X_labels):
            print("Acceptance : ",acc)
            for j, emo in enumerate(label):
                print("\t {} = {} ... {:.3g}%".format(emo, former[i][j],acc_per_emo[i][j]))
    ## acc_per_emo -> emotion size, acceptance size
    acc_per_emo = [[acc_per_emo[acc][emo] for acc in range(len(labels))] for emo in range(len(label))]
    for emo in range(len(label)):
        bottom = [sum([acc_per_emo[e][acc] for e in range(emo)]) for acc in range(len(labels))]
        print(bottom)
        plt.bar(left, acc_per_emo[emo], color=color_list[emo], bottom=bottom, label=label[emo])
        plt.legend(loc='best',shadow=True)
    if not max_y == None:
        plt.ylim(0,max_y)
    plt.xticks(left, labels)


# In[503]:


former_emotion(user_acc3_all, system_emo3_1_all, Plot_type_2 = True, y_label="Number of emotion labels")


# In[504]:


former_emotion(user_acc3_all, system_emo3_2_all,Plot_type_2 = True, y_label="Number of emotion labels")


# ## Former emotion expressed by the system

# In[505]:


former_emotion(user_acc3_all, system_emo3_1_all,Plot_type_2=True, Plot2_ratio=True, max_y = 1.8,
              y_label = "Rate of emotion")


# ## Later emotion expressed by the system

# In[506]:


former_emotion(user_acc3_all, system_emo3_2_all,Plot_type_2=True, Plot2_ratio=True, max_y = 1.8,
              y_label = "Rate of emotion")


# In[60]:


Counter(user_acc3[1])


# 3,4,5を使うべき、説得出来ていない発話を使う意味が分からない
# 
# 絞った時と全部使ったときと比べたらよい
# 
# 上手くいっているのを優先的に使うべき
# 
# その前の部分を見ないと意味ない

# 説得が上手くいっている場合のシステムの発話と対話の流れを見るべき
# 
# 使えそうな発話を割り出したい
# 
# 受諾度に感情が関わっているかの相関を見たほうが良い
# 
# 基の分布と比べてみるべき
# 
# 戦略を作るために、データで効いている感情遷移を
# 
# 

# In[ ]:


len(dialog2_us[0][0])


# In[ ]:


自分でやってみて、
重要な文をランク付けする必要がある
シナリオを作るとしたらオートマトンをゴリゴリ書く

