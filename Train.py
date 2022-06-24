import pandas as pd
import numpy as np
import torch
import re
import os
import gc
import math
import random
import time
import Proto
import Normal
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
# 画图
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

seed_list = list(range(1,11))
experiment_times = len(seed_list)
# de_train和df_test是已经处理好的确定的数据集

df_train = pd.read_csv('./data/Key Indicators/stock2020 Key Indicators training.csv')
df_test = pd.read_csv('./data/Key Indicators/stock2020 Key Indicators test.csv')

# df_train = pd.read_csv('./data/Expanded Indicators/stock2020 Expanded Indicators training.csv')
# df_test = pd.read_csv('./data/Expanded Indicators/stock2020 Expanded Indicators test.csv')


# df_train = pd.read_csv('./data/Filtered Indicators/ROE15 net cash ratio0.9 training.csv')
# df_test = pd.read_csv('./data/Filtered Indicators/ROE15 net cash ratio0.9 test.csv')

df_test_normal = df_test
df_train_normal = df_train
train_size = df_train.shape[0]
test_size = df_test.shape[0]

# 把测试集的label拿出来
test_label = df_test['label'].values
df_test = df_test.drop(columns=['label'])

#acc是准确率，pre是精准率，rec是recall，F1是F1_score，AUC
acc_train_protos = pre_train_protos = rec_train_protos = F1_train_protos = AUC_train_protos = 0.0
acc_valid_protos = pre_valid_protos = rec_valid_protos = F1_valid_protos = AUC_valid_protos = 0.0
acc_test_protos = pre_test_protos = rec_test_protos = F1_test_protos = AUC_test_protos = 0.0

acc_train_protos_total = pre_train_protos_total = rec_train_protos_total = F1_train_protos_total = AUC_train_protos_total = 0.0
acc_valid_protos_total = pre_valid_protos_total = rec_valid_protos_total = F1_valid_protos_total = AUC_valid_protos_total = 0.0
acc_test_protos_total = pre_test_protos_total = rec_test_protos_total = F1_test_protos_total = AUC_test_protos_total = 0.0

acc_train_protob = pre_train_protob = rec_train_protob = F1_train_protob = AUC_train_protob = 0.0
acc_valid_protob = pre_valid_protob = rec_valid_protob = F1_valid_protob = AUC_valid_protob = 0.0
acc_test_protob = pre_test_protob = rec_test_protob = F1_test_protob = AUC_test_protob = 0.0

acc_train_normals = pre_train_normals = rec_train_normals = F1_train_normals = AUC_train_normals = 0.0
acc_valid_normals = pre_valid_normals = rec_valid_normals = F1_valid_normals = AUC_valid_normals = 0.0
acc_test_normals = pre_test_normals = rec_test_normals = F1_test_normals = AUC_test_normals = 0.0

acc_train_normalb = pre_train_normalb = rec_train_normalb = F1_train_normalb = AUC_train_normalb = 0.0
acc_valid_normalb = pre_valid_normalb = rec_valid_normalb = F1_valid_normalb = AUC_valid_normalb = 0.0
acc_test_normalb = pre_test_normalb = rec_test_normalb = F1_test_normalb = AUC_test_normalb = 0.0

acc_train_normalb_total = pre_train_normalb_total = rec_train_normalb_total = F1_train_normalb_total = AUC_train_normalb_total = 0.0
acc_valid_normalb_total = pre_valid_normalb_total = rec_valid_normalb_total = F1_valid_normalb_total = AUC_valid_normalb_total = 0.0
acc_test_normalb_total = pre_test_normalb_total = rec_test_normalb_total = F1_test_normalb_total = AUC_test_normalb_total = 0.0

for i in range(experiment_times):
    # 第几次实验
    print(f'--------第{i+1}次实验开始---------')
    # 确定本次实验使用的种子
    random.seed(seed_list[i])
    # 获得训练集的索引列表
    index_list = list(df_train.index)
    # 打乱索引列表，种子一样的话，打乱后的列表也是相同的
    random.shuffle(index_list)
    # 定义一个空的df，用来保存新顺序的df_train
    df_train_re = pd.DataFrame()
    # df_train_1 表示新的训练集
    for j in index_list:
        df_train_re = df_train_re.append(df_train.loc[j], ignore_index=True)

    # 根据标签区分为两个df，然后随机打乱
    df_0 = df_train[df_train['label'] == 0]
    df_1 = df_train[df_train['label'] == 1]
    index_list_0 = df_0.index.to_list()
    index_list_1 = df_1.index.to_list()
    random.seed(seed_list[i])
    random.shuffle(index_list_0)
    random.seed(seed_list[i])
    random.shuffle(index_list_1)
    df_0_shuffle = pd.DataFrame()
    df_1_shuffle = pd.DataFrame()
    for j in index_list_0:
        df_0_shuffle = df_0_shuffle.append(df_train.loc[i])
    for j in index_list_1:
        df_1_shuffle = df_1_shuffle.append(df_train.loc[i])

    df_0_shuffle = df_0_shuffle.reset_index(drop=True)
    df_1_shuffle = df_1_shuffle.reset_index(drop=True)

    # 把0和1的df的label列拿删除
    # df_0_shuffle = df_0_shuffle.drop(columns=['label'])
    # df_1_shuffle = df_1_shuffle.drop(columns=['label'])

    df_list = [df_0_shuffle, df_1_shuffle]
    # print(df_1.shape[0], df_0.shape[0])


    #################### Protonets小样本 ####################
    # os.system(f'python ./Proto {df_train_1}')
    # 超参数设置
    input_dim_psmall = 11 #精简指标
    # input_dim_psmall = 14 #多指标
    output_dim_psmall = 8 #输出维度
    lr_psmall = 0.02  #学习率
    epoch_size_psmall = 1000 #迭代次数
    support_size_psmall = 5  #support集大小
    query_size_psmall = 2  #query集大小

    protonets_s = Proto.Protonets(support_size_psmall, query_size_psmall, 2, input_dim_psmall, output_dim_psmall, lr_psmall, seed_list[i], epoch_size_psmall)

    acc_train_protos_total_1 = pre_train_protos_total_1 = rec_train_protos_total_1 = F1_train_protos_total_1 = AUC_train_protos_total_1 = 0.0
    for epoch in range(epoch_size_psmall):
        acc_train_protos, pre_train_protos, rec_train_protos, F1_train_protos, AUC_train_protos = protonets_s.train(df_list, epoch)
        acc_train_protos_total_1 += acc_train_protos
        pre_train_protos_total_1 += pre_train_protos
        rec_train_protos_total_1 += rec_train_protos
        F1_train_protos_total_1 += F1_train_protos
        AUC_train_protos_total_1 += AUC_train_protos

    acc_train_protos_total += acc_train_protos_total_1/epoch_size_psmall
    pre_train_protos_total += pre_train_protos_total_1/epoch_size_psmall
    rec_train_protos_total += rec_train_protos_total_1/epoch_size_psmall
    F1_train_protos_total += F1_train_protos_total_1/epoch_size_psmall
    AUC_train_protos_total += AUC_train_protos_total_1/epoch_size_psmall

    acc_test_protos, pre_test_protos, rec_test_protos, F1_test_protos, AUC_test_protos = protonets_s.test(df_test, test_label, '小')
    acc_test_protos_total += acc_test_protos
    pre_test_protos_total += pre_test_protos
    rec_test_protos_total += rec_test_protos
    F1_test_protos_total += F1_test_protos
    AUC_test_protos_total += AUC_test_protos

    protonets_s.save('D:/股价 小样本/原型网络/log2020 proto小样本/')


    ############################# 普通神经网络大样本 ###################################
    batch_size = 5 #批量大小
    k = 5 #k折次数
    lr = 0.02 #学习率
    epoch_size = 6 #迭代次数

    normal_b = Normal.Normalnets(batch_size, k, lr, epoch_size)
    acc_train_normalb, acc_valid_normalb, pre_train_normalb, pre_valid_normalb, rec_train_normalb, rec_valid_normalb, F1_train_normalb, F1_valid_normalb, AUC_train_normalb, AUC_valid_normalb = normal_b.train(df_train_normal)
    acc_train_normalb_total += acc_train_normalb
    acc_valid_normalb_total += acc_valid_normalb
    pre_train_normalb_total += pre_train_normalb
    pre_valid_normalb_total += pre_valid_normalb
    rec_train_normalb_total += rec_train_normalb
    rec_valid_normalb_total += rec_valid_normalb
    F1_train_normalb_total += F1_train_normalb
    F1_valid_normalb_total += F1_valid_normalb
    AUC_train_normalb_total += AUC_train_normalb
    AUC_valid_normalb_total += AUC_valid_normalb

    acc_test_normalb, pre_test_normalb, rec_test_normalb, F1_test_normalb, AUC_test_normalb = normal_b.test(df_test_normal, './logs NormalNet20年报大样本 2022 03 15', '大')
    acc_test_normalb_total += acc_test_normalb
    pre_test_normalb_total += pre_test_normalb
    rec_test_normalb_total += rec_test_normalb
    F1_test_normalb_total += F1_test_normalb
    AUC_test_normalb_total += AUC_test_normalb


    normal_b.save('D:/股价 小样本/原型网络/log2020 normal大样本/')

    if i == experiment_times-1:
        df_result = pd.DataFrame([['Normal',
                                   acc_train_normalb_total / experiment_times,
                                   AUC_train_normalb_total / experiment_times,
                                   pre_train_normalb_total / experiment_times,
                                   rec_train_normalb_total / experiment_times,
                                   F1_train_normalb_total / experiment_times,
                                   acc_valid_normalb_total / experiment_times,
                                   AUC_valid_normalb_total / experiment_times,
                                   pre_valid_normalb_total / experiment_times,
                                   rec_valid_normalb_total / experiment_times,
                                   F1_valid_normalb_total / experiment_times,
                                   acc_test_normalb_total / experiment_times,
                                   AUC_test_normalb_total / experiment_times,
                                   pre_test_normalb_total / experiment_times,
                                   rec_test_normalb_total / experiment_times,
                                   F1_test_normalb_total / experiment_times]],
                                 columns=['模型', '训练Accuracy', '训练AUC', '训练Precision', '训练Recall', '训练F1',
                                          '验证Accuracy', '验证AUC', '验证Precision', '验证Recall', '验证F1', '测试Accuracy',
                                          '测试AUC', '测试Precision', '测试Recall', '测试F1'])
        df_result.to_csv('../其他模型/筛选数据测试结果.csv', mode='a', index=0, header=None)
        print(f'Protonets小样本训练平均准确率为：{acc_train_protos_total / experiment_times}，精准率：{pre_train_protos_total / experiment_times}， '
              f'召回率：{rec_train_protos_total / experiment_times}， F1：{F1_train_protos_total / experiment_times}， '
              f'AUC：{AUC_train_protos_total / experiment_times}\n'
              f'测试平均准确率为：{acc_test_protos_total / experiment_times}，AUC：{AUC_test_protos_total / experiment_times}， 精准率：{pre_test_protos_total / experiment_times}， '
              f'召回率：{rec_test_protos_total / experiment_times}， F1：{F1_test_protos_total / experiment_times}'
              f'')
        # print(f'Protonets大样本平均准确率为：{accuracy_protob / experiment_times}')
        print(f'Normalnets大样本训练平均准确率为：{acc_train_normalb_total / experiment_times}，AUC：{AUC_train_normalb_total / experiment_times}, '
              f'精准率：{pre_train_normalb_total / experiment_times}， '
              f'召回率：{rec_train_normalb_total / experiment_times}，F1：{F1_train_normalb_total / experiment_times}\n'
              f'验证平均准确率为：{acc_valid_normalb_total / experiment_times}，AUC：{AUC_valid_normalb_total / experiment_times}，精准率：{pre_valid_normalb_total / experiment_times}， '
              f'召回率：{rec_valid_normalb_total / experiment_times}，F1：{F1_valid_normalb_total / experiment_times}\n'
              f'测试平均准确率为：{acc_test_normalb_total / experiment_times}，AUC：{AUC_test_normalb_total / experiment_times}，精准率：{pre_test_normalb_total / experiment_times}，'
              f'召回率：{rec_test_normalb_total / experiment_times}，F1：{F1_test_normalb_total / experiment_times}')
        # print(f'Normalnets小样本平均准确率为：{accuracy_normals / experiment_times}')