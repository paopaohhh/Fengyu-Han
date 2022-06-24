# 普通的多层感知机
import pandas as pd
import numpy as np
import torch
import re
import gc
import math
import random
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
# 数据加载
class mydataloader(Dataset):
    def __init__(self, dataset, str:str):
        self.label = dataset[str]
        self.data = dataset.drop(str, axis = 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        label = torch.tensor(self.label[index])
        data = torch.tensor(self.data.iloc[index])
        data = data.to(torch.float32)
        return data, label

'''class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2))
        self.net2 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2))
        # self.net3 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5))
        self.net3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        # x = self.net4(x)
        return x'''

class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(input_dim, 64), nn.BatchNorm1d(64),nn.ReLU(), nn.Dropout(0.5))
        self.net2 = nn.Sequential(nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5))
        # self.net3 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5))
        self.net3 = nn.Sequential(nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        # x = self.net4(x)
        return x


class Normalnets(object):
    def __init__(self, batch_size, k, lr, epoch_size):
        self.batch_size = batch_size
        self.k = k
        self.lr = lr
        self.epoch_size = epoch_size
        self.net = Net(11, 2) #精简指标
        # self.net = Net(14, 2) #多指标
        # self.net = self.net.cuda()
        self.loss = nn.CrossEntropyLoss()
        # self.loss = self.loss.cuda()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.acc_train_total = 0.0
        self.acc_valid_total = 0.0
        self.pre_train_total = 0.0
        self.pre_valid_total = 0.0
        self.rec_train_total = 0.0
        self.rec_valid_total = 0.0
        self.AUC_train_total = 0.0
        self.AUC_valid_total = 0.0
        self.F1_train_total = 0.0
        self.F1_valid_total = 0.0
        self.acc_test_total = 0.0
        self.pre_test_total = 0.0
        self.rec_test_total = 0.0
        self.AUC_test_total = 0.0
        self.F1_test_total = 0.0

    ########k折划分############
    def get_k_fold_data(self, k, i, X):  ###X是dataframe
        # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
        assert k > 1
        # print(X.shape[0])
        fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

        X_train, y_train = None, None
        for j in range(k):
            idx = list(range(j * fold_size, (j + 1) * fold_size))  # slice(start,end,step)切片函数
            ##idx 为每组 valid
            X_part = X.loc[idx]
            if j == i:  ###第i折作valid
                X_valid = X_part
            elif X_train is None:
                X_train = X_part
            else:
                X_train = pd.concat([X_train, X_part])  # dim=0增加行数，竖着连接
        # print(X_train.size(),X_valid.size())
        return X_train, X_valid

    def train(self, df_train):
        self.train_size = df_train.shape[0]
        self.valid_size = self.train_size // self.k
        writer = SummaryWriter('./logs NormalNet20年报大样本 2022 03 15')

        # F1_train = 2 * pre_train * rec_train / (pre_train + rec_train)
        # F1_valid = 2 * pre_valid * rec_valid / (pre_valid + rec_valid)
        acc_train_total = 0.0
        acc_valid_total = 0.0
        pre_train_total = 0.0
        pre_valid_total = 0.0
        rec_train_total = 0.0
        rec_valid_total = 0.0
        AUC_train_total = 0.0
        AUC_valid_total = 0.0
        F1_train_total = 0.0
        F1_valid_total = 0.0


        for i in range(self.k):
            train_data, valid_data = self.get_k_fold_data(self.k, i, df_train)
            train_data = train_data.reset_index(drop=True)
            valid_data = valid_data.reset_index(drop=True)
            train_set = mydataloader(train_data, 'label')
            valid_set = mydataloader(valid_data, 'label')
            # train_set = mydataloader(train_data, 'y')
            # valid_set = mydataloader(valid_data, 'y')
            train_iter = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
            valid_iter = DataLoader(valid_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
            # 定义训练步骤和测试步骤
            train_step = 0
            valid_step = 0
            train_size = train_data.shape[0]
            valid_size = valid_data.shape[0]

            acc_train = 0.0
            acc_valid = 0.0
            pre_train = 0.0
            pre_valid = 0.0
            rec_train = 0.0
            rec_valid = 0.0
            AUC_train = 0.0
            AUC_valid = 0.0
            F1_train = 0.0
            F1_valid = 0.0
            # 开始训练
            for epoch in range(self.epoch_size):

                y_true_train = torch.tensor([])
                y_pre_train = torch.tensor([])
                y_true_valid = torch.tensor([])
                y_pre_valid = torch.tensor([])
                # print(f"----第{i + 1}折，第{epoch + 1}轮训练开始----")
                for data in train_iter:
                    # train_times += 1
                    x, y = data
                    # x = x.cuda()
                    # y = y.cuda()
                    y_hat = self.net(x)
                    l = self.loss(y_hat, y.long())
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    train_step += 1
                    y_hat_arg = y_hat.argmax(1)
                    # result = y_hat.argmax(1) == y
                    # accuracy = result.sum()
                    # for i in range(len(y)):
                    #     if y[i] == 0:
                    #         if y_hat[i].values == 1:
                    #             FP_train+=1
                    #         else:
                    #             TN_train+=1
                    #     else:
                    #         if y_hat[i].values == 1:
                    #             TP_train+=1
                    #         else:
                    #             FN_train+=1
                    y_true_train = torch.concat((y_true_train ,y.detach().view(5)))
                    y_pre_train = torch.concat((y_pre_train, y_hat.detach().argmax(1)))


                    # total_accuracy += accuracy
                    writer.add_scalar("Train", l, train_step)
                    # if train_step % 100 == 0:
                        # print(f"第{train_step}次训练，loss：{l}")
                # print(f"第{i + 1}折，第{train_step}次训练，准确率：{total_accuracy / train_size}")
                # if i == self.k-1:
                #     print(f'训练准确率为：{total_accuracy/train_size}')


                l_sum = 0
                # 求精准度
                # acc_train += total_accuracy/train_size
                # total_accuracy = 0.0
                # print(f"----第{i + 1}折，第{epoch + 1}轮验证开始----")

                for data in valid_iter:
                    # valid_times += 1
                    x, y = data
                    # x = x.cuda()
                    # y = y.cuda()
                    y_hat = self.net(x)
                    l = self.loss(y_hat, y.long())
                    l_sum += l
                    l_valid = l
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    valid_step += 1
                    # 求精准度
                    y_hat_arg = y_hat.argmax(1)
                    # accuracy = (y_hat.argmax(1) == y).sum()
                    # for i in range(len(y)):
                    #     if y[i] == 0:
                    #         if y_hat_arg[i].values == 1:
                    #             FP_valid += 1
                    #         else:
                    #             TN_valid += 1
                    #     else:
                    #         if y_hat_arg[i].values == 1:
                    #             TP_valid += 1
                    #         else:
                    #             FN_valid += 1

                    writer.add_scalar("Valid", l, valid_step)
                    # total_accuracy += accuracy
                    y_true_valid = torch.concat((y_true_valid, y.detach().view(5)))
                    y_pre_valid = torch.concat((y_pre_valid, y_hat.detach().argmax(1)))
                # print(f"第{i + 1}折，第{epoch + 1}轮验证，loss：{l_sum / valid_size}, 准确率为：{total_accuracy / valid_size}")
                # if i == self.k-1:
                #     print(f'验证准确率为：{total_accuracy/valid_size}')
                # acc_valid += total_accuracy/valid_size

                # acc_train += (TP_train + TN_train) / (TP_train + TN_train + FP_train + FN_train)
                # acc_valid += (TP_valid + TN_valid) / (TP_valid + TN_valid + FP_valid + FN_valid)
                # pre_train += TP_train / (TP_train + FP_train)
                # pre_valid += TP_valid / (TP_valid + FP_valid)
                # rec_train += TP_train / (TP_train + FN_train)
                # rec_valid += TP_valid / (TP_valid + FN_valid)
                # F1_train += 2 * pre_train * rec_train / (pre_train + rec_train)
                # F1_valid += 2 * pre_valid * rec_valid / (pre_valid + rec_valid)

                y_true_train = y_true_train.numpy()
                y_pre_train = y_pre_train.numpy()
                y_true_valid = y_true_valid.numpy()
                y_pre_valid = y_pre_valid.numpy()

                acc_train += accuracy_score(y_true_train, y_pre_train)
                acc_valid += accuracy_score(y_true_valid, y_pre_valid)
                pre_train += precision_score(y_true_train, y_pre_train)
                pre_valid += precision_score(y_true_valid, y_pre_valid)
                rec_train += recall_score(y_true_train, y_pre_train)
                rec_valid += recall_score(y_true_valid, y_pre_valid)
                F1_train += f1_score(y_true_train, y_pre_train)
                F1_valid += f1_score(y_true_valid, y_pre_valid)
                AUC_train += roc_auc_score(y_true_train, y_pre_train)
                AUC_valid += roc_auc_score(y_true_valid, y_pre_valid)


            acc_train_total += acc_train / self.epoch_size
            acc_valid_total += acc_valid / self.epoch_size
            pre_train_total += pre_train / self.epoch_size
            pre_valid_total += pre_valid / self.epoch_size
            rec_train_total += rec_train_total / self.epoch_size
            rec_valid_total += rec_train_total / self.epoch_size
            F1_train_total += F1_train / self.epoch_size
            F1_valid_total += F1_valid / self.epoch_size
            AUC_train_total += AUC_train / self.epoch_size
            AUC_valid_total += AUC_valid / self.epoch_size


        self.acc_train_total = acc_train_total / self.k
        self.acc_valid_total = acc_valid_total / self.k
        self.pre_train_total = pre_train_total / self.k
        self.pre_valid_total = pre_valid_total / self.k
        self.rec_train_total = rec_train_total / self.k
        self.rec_valid_total = rec_valid_total / self.k
        self.AUC_train_total = AUC_train_total / self.k
        self.AUC_valid_total = AUC_valid_total / self.k
        self.F1_train_total = F1_train_total / self.k
        self.F1_valid_total = F1_valid_total / self.k

        return self.acc_train_total, self.acc_valid_total, self.pre_train_total, self.pre_valid_total, self.rec_train_total, \
               self.rec_valid_total, self.F1_train_total, self.F1_valid_total, self.AUC_train_total, self.AUC_valid_total


    def test(self, df_test, path, str_0):
        self.test_size = df_test.shape[0]
        acc_test = 0.0
        pre_test = 0.0
        rec_test = 0.0
        AUC_test = 0.0
        F1_test = 0.0

        TP_test = 0.0
        TN_test = 0.0
        FP_test = 0.0
        FN_test = 0.0
        # 测试集的迭代器
        test_set = mydataloader(df_test, 'label')
        # test_set = mydataloader(df_test, 'y')
        test_iter = DataLoader(test_set, batch_size=self.batch_size, shuffle=True, drop_last=True)
        writer = SummaryWriter(path)
        # './logs NormalNet20年报大样本 2022 03 15'
        scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score), "Precision": make_scorer(precision_score),
                   "Recall": make_scorer(recall_score), "F1": make_scorer(f1_score)}
        l_sum = 0
        # 求精准度
        total_accuracy = 0.0
        test_step = 0
        # print(f"----第{epoch + 1}轮测试开始----")
        l_last = 0
        test_times = 0
        y_true_test = torch.tensor([])
        y_pre_test = torch.tensor([])
        with torch.no_grad():
            for data in test_iter:
                test_times += 1
                x, y = data
                y_hat = self.net(x)
                l = self.loss(y_hat, y.long())
                l_last = l
                l_sum += l
                test_step += 1
                # 求精准度
                y_hat_arg = y_hat.argmax(1)
                bool_accuray = y_hat.argmax(1) == y
                accuracy = bool_accuray.sum()
                # if accuracy == 0:
                #     if y == 1:
                #         FN_test += 1
                #     else:
                #         FP_test += 1
                # else:
                #     if y == 1:
                #         TP_test += 1
                #     else:
                #         TN_test += 1

                y_true_test = torch.concat((y_true_test, y.detach().view(5)))
                y_pre_test = torch.concat((y_pre_test, y_hat.detach().argmax(1)))

                writer.add_scalar("Test", l, test_step)
                total_accuracy += accuracy
            self.accuracy = total_accuracy / float(self.test_size)
            print('普通神经网络' + str_0 +f'样本测试准确率为：{self.accuracy}')

        y_true_test = y_true_test.numpy()
        y_pre_test = y_pre_test.numpy()
        self.acc_test_total = accuracy_score(y_true_test, y_pre_test)
        self.pre_test_total = precision_score(y_true_test, y_pre_test)
        self.rec_test_total = recall_score(y_true_test, y_pre_test)
        self.F1_test_total = f1_score(y_true_test, y_pre_test)
        self.AUC_test_total = roc_auc_score(y_true_test, y_pre_test)

        return self.acc_test_total, self.pre_test_total, self.rec_test_total, self.F1_test_total, self.AUC_test_total

        writer.close()

    def save(self, path): # 'D:/股价 小样本/原型网络/log2020 normal大样本/'
        # 储存模型
        torch.save(
            {'model': self.net,
             'lr': self.lr,
             'k': self.k,
             'batch_szie': self.batch_size,
             'epoch_size': self.epoch_size,
             'train_size': self.train_size,
             'test_size': self.test_size},
            path + f'{self.accuracy:.2f} model_net_' + str(
                time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())) + '.pkl')