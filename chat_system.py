from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
import collections
import json
import re
import os
import pprint
import numpy as np
import tensorflow as tf
import modeling
import tokenization
import socket
import logging

from extract_features import InputExample, InputFeatures
from extract_features import input_fn_builder, model_fn_builder, convert_examples_to_features, _truncate_seq_pair
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import  Counter
import sentencepiece as spm
import argparse
import pickle

BERT_PRETRAINED_DIR = "../trained_model/masked_lm_only_L-12_H-768_A-12"
LAYERS = [-1]
NUM_TPU_CORES = 8
MAX_SEQ_LENGTH = 512
BERT_CONFIG = BERT_PRETRAINED_DIR+"/bert_config.json"
CHKPT_DIR = BERT_PRETRAINED_DIR+"/model.ckpt-1000000"
INIT_CHECKPOINT = BERT_PRETRAINED_DIR+"/model.ckpt-1000000"
VOCAB_FILE=BERT_PRETRAINED_DIR+"/tokenizer_spm_32K.vocab.to.bert"
VOCAB_MODEL=BERT_PRETRAINED_DIR+"/tokenizer_spm_32K.model"
BATCH_SIZE = 128


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir",
                        help="data directory",
                        default="text_data")
    parser.add_argument("-e", "--emotion",
                        help="system's emotion",
                        default="neutral")
    parser.add_argument("-t", "--total_num",
                        help="total of the utterances during chat",
                        default=20)
    parser.add_argument("-l", "--use_local",
                        help="connect local machine",
                        action="store_true")
    parser.add_argument("-m", "--model_path",
                        help="model path to select emotion",
                        default="emotion_model/model/lr_model_2emo.sav")
    return parser

def read_sequence(input_sentences):
    examples = []
    unique_id = 0
    for sentence in input_sentences:
        line = tokenization.convert_to_unicode(sentence)
        examples.append(InputExample(unique_id=unique_id, text_a=line, text_b=None))
        unique_id += 1
    return examples

def get_features(input_text, dim=768, code_num=1):
    tf.logging.set_verbosity(tf.logging.ERROR)
    layer_indexes = LAYERS
    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
    _normalizer = None
    tokenizer = tokenization.JapaneseTweetTokenizer(
        vocab_file=VOCAB_FILE,
        model_file=VOCAB_MODEL,
        normalizer=_normalizer,
        do_lower_case=False)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        master=None,
        tpu_config=tf.contrib.tpu.TPUConfig(
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=is_per_host))
    
    examples = read_sequence(input_text)

    features = convert_examples_to_features(
          examples=examples, seq_length=MAX_SEQ_LENGTH, tokenizer=tokenizer)

    unique_id_to_feature = {}
    for feature in features:
        unique_id_to_feature[feature.unique_id] = feature

    model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=INIT_CHECKPOINT,
      layer_indexes=layer_indexes,
      use_tpu=False,
      use_one_hot_embeddings=False)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=False,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=BATCH_SIZE)

    input_fn = input_fn_builder(
        features=features, seq_length=MAX_SEQ_LENGTH)

    # Get features
    if code_num == 1:
        for result in estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            layer_output = result["layer_output_0"]
            layer_output_flat = np.array([x for x in layer_output[0].flat])
            output = layer_output_flat[:dim]
    else:
        output = np.array([result["layer_output_0"][0] for result in estimator.predict(
        input_fn, yield_single_examples=True)])[-1][:dim]
    return output

def calc_similarity(train_dial, test_dial, model=None, Euclidian=True):
    train_dial = np.array(train_dial)
    test_dial = np.array(test_dial)
    if Euclidian == False:
        similarity = cosine_similarity(train_dial, test_dial)    
    else:
        similarity = -euclidean_distances(train_dial, test_dial, Y_norm_squared= pow(test_dial,2).sum(axis=1),squared=True)
    sum_similarity = np.array([sum(i) for i in similarity]) 
    top_index = np.argsort(sum_similarity)[::-1][0]
    cos_sim = np.sort(sum_similarity)[::-1][0]
    return int(top_index), cos_sim


def chat(input_text, emotion="neutral", dir_path="text_data", output_log=False):
    np_user1 =np.loadtxt(dir_path+"/user1_dial.tsv")
    np_system2 =np.loadtxt(dir_path+"/"+emotion+"_dial.tsv")
    user1_embedding = get_features([input_text])
    sim_id = calc_similarity(np.array(np_user1), np.array([user1_embedding.tolist()])) 
    
    user1_uttrs = open(dir_path+"/user1_dial.txt","r").readlines()
    system2_uttrs = open(dir_path+"/"+emotion+"_dial.txt","r").readlines()
    user1_uttr = user1_uttrs[sim_id].strip("\n")
    system2_uttr = system2_uttrs[sim_id].strip("\n")
    system2_embedding = np_system2[sim_id]
    #print("\n    -> (類似度の高かったユーザ発話：{})".format(user1_uttr))
    print("\nロボ：{}".format(system2_uttr))
    return sim_id, system2_embedding


def chat2(input_text, pre_id, system2_embedding=None, emotion="neutral",
          dir_path="text_data", output_log=False):
    np_user1 =np.loadtxt(dir_path+"/user1_dial.tsv")
    np_system1 =np.loadtxt(dir_path+"/system1_dial.tsv")
    np_system2 =np.loadtxt(dir_path+"/"+emotion+"_dial.tsv")
    user2_embedding = get_features([input_text])
    sim_id, cos_sim = calc_similarity(np.concatenate([np.array(np_system1),np.array(np_user1)],axis=1),
                             np.concatenate([np.array([np_system2[pre_id]]),
                                             np.array([user2_embedding.tolist()])],axis=1))
    system1_uttrs = open(dir_path+"/system1_dial.txt","r").readlines()
    user1_uttrs = open(dir_path+"/user1_dial.txt","r").readlines()
    system2_uttrs = open(dir_path+"/"+emotion+"_dial.txt","r").readlines()
    
    system1_uttr = system1_uttrs[sim_id].strip("\n")
    user1_uttr = user1_uttrs[sim_id].strip("\n")
    system2_uttr = system2_uttrs[sim_id].strip("\n")
    system2_embedding = np_system2[sim_id]
    print("     (類似度の高かった発話ペア...ロボ: {}\n     \t\t\t   ユーザ: {})".format(system1_uttr,user1_uttr))
    print("ロボ：{}".format(system2_uttr))
    return sim_id, cos_sim, user2_embedding, system2_embedding

def chat3(input_text, pre_id, emotion="neutral",
          dir_path="text_data", print_uttrs=True, pre_emotion=None, emo_list=None): 

    user2_embedding = get_features([input_text])    
    first_emotion = emotion
    
    if emotion == "nothing":
        emotion, first_emotion = select_emotion(pre_emotion, user2_embedding, emo_list)
        print("previous emotion : {}\nnext emotion : {}".format(pre_emotion, emotion))
    
    # load sentence vectors in corpus
    np_user1 =np.loadtxt(dir_path+"/user1_dial.tsv")
    np_system1 =np.loadtxt(dir_path+"/system1_dial.tsv")
    np_system2 =np.loadtxt(dir_path+"/"+emotion+"_dial.tsv")
        
    sim_id, cos_sim = calc_similarity(np.concatenate([np.array(np_system1),np.array(np_user1)],axis=1),
                             np.concatenate([np.array([np_system2[pre_id]]),
                                             np.array([user2_embedding.tolist()])],axis=1))  
    if print_uttrs:
        system1_uttrs = open(dir_path+"/system1_dial.txt","r").readlines()
        user1_uttrs = open(dir_path+"/user1_dial.txt","r").readlines()
        system2_uttrs = open(dir_path+"/"+emotion+"_dial.txt","r").readlines()
        system1_uttr = system1_uttrs[sim_id].strip("\n")
        user1_uttr = user1_uttrs[sim_id].strip("\n")
        system2_uttr = system2_uttrs[sim_id].strip("\n")
        system2_embedding = np_system2[sim_id]
        print("(類似度の高かったペア...ロボ: {}\n\t\t   ユーザ: {})".format(
              system1_uttr,user1_uttr))
        print("ロボ：{}".format(system2_uttr))
    return sim_id, cos_sim, emotion, first_emotion


def chat_system(chat_num, emotion, dir_path, total_num=20, select_emo=False):
    print("ロボ：最近運動不足だしジョギングしようよ")
    for i in range(total_num):
        input_text = input("ユーザ：")
        if input_text == "exit" or ("分かった" or "いいよ" or "仕方ないな" or "了解" or "わかった") in input_text:
            print("ロボ：分かってくれてありがとう！")
            break
        if i==0 or chat_num==1:
            pre_id, system2_embedding = chat(input_text, emotion=emotion, dir_path=dir_path)
        else:
            pre_id, user2_embedding, system2_embedding = chat2(
                input_text, system2_embedding=system2_embedding, emotion=emotion, dir_path=dir_path)

            
def select_emotion(pre_emotion, user_embedding, emo_list):
    id2emo = {0:"neutral",1:"anger",2:"sad",3:"happy"}
    emo_vector_dic2 = {"neutral":[1,0,0,0],"anger":[0,1,0,0],
                       "sad":[0,0,1,0], "happy":[0,0,0,1]}
    model = pickle.load(open(model_path, "rb"))
    input_vector = np.array([np.concatenate(
        [np.array(emo_vector_dic2[pre_emotion]),user_embedding])])
    first_emotion = pre_emotion
    
    emotion_id = model.predict(input_vector)[0]
    
    if id2emo[emotion_id] == Counter(emo_list).most_common(1)[0][0] and Counter(emo_list).most_common(1)[0][1] == 3:
        print(id2emo[emotion_id] + "->")
        emotion_id = np.argsort(model.predict_proba(input_vector))[0][::-1][1]
        print(model.predict_proba(input_vector), emotion_id)
        print(id2emo[emotion_id])
        
    emotion = id2emo[emotion_id]
    return emotion, first_emotion 
            
            
if __name__ == '__main__':
    args = get_argparse().parse_args()
    data_dir = args.data_dir
    emotion_name = args.emotion
    total_num = args.total_num
    use_local = args.use_local
    model_path = args.model_path
    
    if use_local:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("0.0.0.0", 50011))
        server.listen(1)    
        emo_list = ["","",""]
        while True:
            conn, addr = server.accept()
            with conn:
                print("Success in connection")
                # Wait untill connection
                while True:
                    # If someone access, receive connection and address
                    data = conn.recv(1024)
                    data = data.decode("utf-8")
                    
                    if not data:
                        break
                    input_text = data.split("\n")[0]
                    pre_id = int(data.split("\n")[1])
                    emotion = data.split("\n")[2]
                    pre_emotion = data.split("\n")[3]
                    num_uttr = int(data.split("\n")[4])
                    print("recognize : ", input_text)
                    print("robot emotion : ", emotion)
                    # Calcurate similarity between input and dialog histories
                    
                    sim_id, cos_sim, emotion, first_emotion = chat3(input_text, pre_id, emotion=emotion, 
                                            pre_emotion=pre_emotion, emo_list=emo_list)
                    
                    if num_uttr == 1:
                        emo_list = ["","",""]
                        emo_id = 0
                        
                    emo_list[emo_id%3] = emotion
                    print(emo_list)
                    emo_id += 1
                    
                    # Return data
                    send_data = str(sim_id) + "\n" + str(cos_sim) + "\n" + emotion + "\n" + first_emotion 
                    encoded_sim_id = send_data.encode(encoding="shift-jis")
                    conn.send(encoded_sim_id)
        
    else:    
        chat_system(2, emotion_name, data_dir, total_num=total_num)
    
    



  
