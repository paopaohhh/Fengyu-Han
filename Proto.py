# 原型网络
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
# 画图
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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


# 样本一共有13个特征，不知道能否使用卷积，可能存在过拟合
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.5))
        self.net2 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5))
        # self.net3 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5))
        self.net3 = nn.Sequential(nn.Linear(64, output_dim))

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        # x = self.net4(x)
        return x

'''class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.net1 = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.5))
        self.net2 = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Dropout(0.5))
        self.net3 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.5))
        self.net4 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.5))
        self.net5 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5))
        self.net6 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5))
        self.net7 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5))
        self.net8 = nn.Sequential(nn.Linear(32, output_dim))

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        x = self.net3(x)
        x = self.net4(x)
        x = self.net5(x)
        x = self.net6(x)
        x = self.net7(x)
        x = self.net8(x)

        return x'''

# 定义要用到的函数
class Protonets(object):
    def __init__(self, Ns, Nq, Nc, input_dim, output_dim, lr_psmall, seed, epoch_size):
        self.Ns = Ns
        self.Nq = Nq
        self.Nc = Nc
        self.lr = lr_psmall
        self.seed = seed
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epoch_size = epoch_size
        self.model = Net(input_dim, output_dim)
        self.center = {}
        self.acc_train_total = 0.0
        self.pre_train_total = 0.0
        self.rec_train_total = 0.0
        self.AUC_train_total = 0.0
        self.F1_train_total = 0.0
        self.acc_test_total = 0.0
        self.pre_test_total = 0.0
        self.rec_test_total = 0.0
        self.AUC_test_total = 0.0
        self.F1_test_total = 0.0

    # 随机采样，返回support set和query set，是datafeame
    def random_sample(self, df):
        index_list = df.index.to_list()
        # 打乱index列表
        random.seed(self.seed)
        random.shuffle(index_list)
        support_index = index_list[:self.Ns]
        query_index = index_list[self.Ns: (self.Ns + self.Nq)]
        support_set = df.loc[support_index]
        query_set = df.loc[query_index]
        return support_set, query_set

    # 计算两个tensor的欧氏距离，用于loss的计算，并且取负
    def eucli_tensor(self, x, y):
        return -1 * torch.sqrt(torch.sum((x - y) * (x - y))).view(1)

    # 标准欧氏距离
    def Standardized_Euclidean_Distance(self, a, b):
        x = a.detach().numpy()
        y = b.detach().numpy()
        X = np.vstack([x, y])

        # 方法一：根据公式求解
        sk = np.var(X, axis=0, ddof=1)
        d1 = np.sqrt([((x - y) ** 2 / sk).sum()])
        d1 = torch.tensor(d1, dtype=torch.float32)
        return d1

    # 计算原点，输入df和Ns，返回tensor
    def compute_center(self, support_set):
        center = 0
        support_set_index = support_set.index.to_list()
        k = 0
        for i in support_set_index:
            data = self.model(torch.tensor(support_set.loc[i]).to(torch.float32))
            if k == 0:
                center = data
            else:
                center += data
            k += 1
        center /= self.Ns
        return center

    # df_list是df_0和df_1组合成的df列表
    def train(self, df_list, step):
        query_set_list = []
        center = {}
        train_result = np.array([])
        y_true_train = np.array([])
        for i in range(self.Nc):
            df = df_list[i]
            # 从df随机取支持集和查询集，依然是df
            support_set, query_set = self.random_sample(df)
            y_true_train = np.append(y_true_train, query_set['label'].values)
            # 丢弃label
            support_set = support_set.drop(columns= 'label')
            query_set = query_set.drop(columns= 'label')
            # 计算中心点
            center[i] = self.compute_center(support_set).view([self.output_dim])
            self.center[i] = center[i]
            # query_set形成一个列表
            for j in query_set.index.to_list():
                query_set_list.append(query_set.loc[j])
                # 优化器
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        optimizer.zero_grad()

        loss_1 = torch.FloatTensor([0])
        for j in range(2):
            for i in range(self.Nc):
                query_set = query_set_list[i+j]
                data = self.model(torch.tensor(query_set).to(torch.float32))
                data = data.view([self.output_dim])
                predict = 0
                for z in range(self.Nc):
                    if z == 0:
                        # predict = self.eucli_tensor(data, self.center[z])
                        # predict = torch.cosine_similarity(data.reshape(data.shape[0],1), self.center[z].reshape(self.center[z].shape[0],1))
                        predict = self.Standardized_Euclidean_Distance(data.reshape(data.shape[0], 1),
                                                                       self.center[z].reshape(self.center[z].shape[0], 1))
                    else:
                        # predict = torch.cat((predict, self.eucli_tensor(data, self.center[z])))
                        # predict = torch.cat((predict, torch.cosine_similarity(data.reshape(data.shape[0],1), self.center[z].reshape(self.center[z].shape[0],1))))
                        #         log_softmax_1 = -1 * self.log_softmax(predict)
                        predict = torch.cat((predict, self.Standardized_Euclidean_Distance(data, self.center[z])))
                loss_1 += -1 * self.log_softmax(predict)[i]

                percentage = F.softmax(predict, dim=0)

                if percentage[0] > percentage[1]:
                    train_result = np.append(train_result, 0)
                else:
                    train_result = np.append(train_result, 1)

                # if percentage[0] > percentage[1]:
                #     train_result.append(0)
                # else:
                #     train_result.append(1)


        loss_1 /= self.Nc * self.Nq

        protonets_loss = loss_1
        protonets_loss.requires_grad_(True)
        protonets_loss.backward()
        optimizer.step()
        y_pre_train = train_result

        self.acc_train_total = accuracy_score(y_true_train, y_pre_train)
        self.pre_train_total = precision_score(y_true_train, y_pre_train)
        self.rec_train_total = recall_score(y_true_train, y_pre_train)
        self.F1_train_total = f1_score(y_true_train, y_pre_train)
        # self.AUC_train_total = roc_auc_score(y_true_train, y_pre_train)

        return self.acc_train_total, self.pre_train_total, self.rec_train_total, self.F1_train_total, self.AUC_train_total


        '''# 绘图
        x = data1_loss[:, 0]
        y = data1_loss[:, 1]
        x1 = data2_loss[:, 0]
        y1 = data2_loss[:, 1]
        fig = plt.figure(figsize=(7, 5))  # figsize是图片的大小`
        ax1 = fig.add_subplot(1, 1, 1)  # ax1是子图的名字`
        # 画出整体loss
        pl.plot(x, y, 'g-', label=u'Dense_Unet(block layer=5)')
        # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
        p2 = pl.plot(x1, y1, 'r-', label=u'RCSCA_Net')
        pl.legend()
        # 显示图例
        p3 = pl.plot(x2, y2, 'b-', label=u'SCRCA_Net')
        pl.legend()
        pl.xlabel(u'iters')
        pl.ylabel(u'loss')
        plt.title('Compare loss for different models in training')
        # 显示放大部分曲线
        # plot the box
        tx0 = 0
        tx1 = 10000
        # 设置想放大区域的横坐标范围
        ty0 = 0.000
        ty1 = 0.12
        # 设置想放大区域的纵坐标范围
        sx = [tx0, tx1, tx1, tx0, tx0]
        sy = [ty0, ty0, ty1, ty1, ty0]
        pl.plot(sx, sy, "purple")
        axins = inset_axes(ax1, width=1.5, height=1.5, loc='right')
        # loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"
        axins.plot(x1, y1, color='red', ls='-')
        axins.plot(x2, y2, color='blue', ls='-')
        axins.axis([0, 20000, 0.000, 0.12])
        plt.savefig("train_results_loss.png")
        pl.show
        # pl.show()也可以'''


    def loss(self, query_set_list):  # 损失函数
        loss_1 = torch.FloatTensor([0])
        for i in range(self.Nc):
            query_set = query_set_list[i]
            data = self.model(torch.tensor(query_set).to(torch.float32))
            data = data.view([self.output_dim])
            predict = 0
            for z in range(self.Nc):
                if z == 0:
                    # predict = self.eucli_tensor(data, self.center[z])
                    # predict = torch.cosine_similarity(data.reshape(data.shape[0],1), self.center[z].reshape(self.center[z].shape[0],1))
                    predict = self.Standardized_Euclidean_Distance(data.reshape(data.shape[0], 1),
                                                      self.center[z].reshape(self.center[z].shape[0], 1))
                else:
                    # predict = torch.cat((predict, self.eucli_tensor(data, self.center[z])))
                    # predict = torch.cat((predict, torch.cosine_similarity(data.reshape(data.shape[0],1), self.center[z].reshape(self.center[z].shape[0],1))))
                    #         log_softmax_1 = -1 * self.log_softmax(predict)
                    predict = torch.cat((predict, self.Standardized_Euclidean_Distance(data, self.center[z])))
            loss_1 += -1 * self.log_softmax(predict)[i]
        loss_1 /= self.Nc * self.Nq
        return loss_1

    # 协助计算损失函数
    def log_softmax(self, predict):
        a = -1 * predict[0]
        b = -1 * predict[1]
        c = math.exp(a)/(math.exp(a) + math.exp(b))
        d = math.exp(b)/(math.exp(a) + math.exp(b))
        c = math.log(c)
        d = math.log(d)
        predict[0] = c
        predict[1] = d
        return predict

    def test(self, df_test, test_label, str_size):
        # 进行测试
        test_result = np.array([])
        y_true_test = test_label
        y_pre_test = np.array([])
        self.test_size = df_test.shape[0]
        for j in df_test.index.to_list():
            data = self.model(torch.FloatTensor(df_test.loc[j]))
            predict = 0
            for z in range(2):
                if z == 0:
                    # predict = protonets.eucli_tensor(data, protonets.center[z])
                    # predict = torch.cosine_similarity(data.reshape(data.shape[0],1), protonets.center[z].reshape(protonets.center[z].shape[0],1))
                    predict = self.Standardized_Euclidean_Distance(data.reshape(data.shape[0], 1),
                                                                          self.center[z].reshape(
                                                                              self.center[z].shape[0], 1))
                else:
                    # predict = torch.cat((predict, self.eucli_tensor(data, self.center[z])))
                    # predict = torch.cat((predict, protonets.eucli_tensor(data, protonets.center[z])))
                    predict = torch.cat(
                        (predict, self.Standardized_Euclidean_Distance(data.reshape(data.shape[0], 1),
                                                                              self.center[z].reshape(
                                                                                  self.center[z].shape[
                                                                                      0], 1))))
            percentage = F.softmax(predict, dim=0)

            # if percentage[0] > percentage[1]:
            #     test_result.append(0)
            # else:
            #     test_result.append(1)

            if percentage[0] > percentage[1]:
                test_result = np.append(test_result, 0)
            else:
                test_result = np.append(test_result, 1)

        # num_right = 0
        # for j in range(self.test_size):
        #     if test_label[j] == test_result[j]:
        #         num_right += 1
        # self.accuracy = float(num_right) / float(self.test_size)
        # print('Protonets'+ str_size +'样本测试准确率为' + str(self.accuracy))
        # return self.accuracy
        self.y_true_test = y_true_test
        self.y_pre_test = y_pre_test = test_result

        self.acc_test_total = accuracy_score(y_true_test, y_pre_test)
        self.pre_test_total = precision_score(y_true_test, y_pre_test)
        self.rec_test_total = recall_score(y_true_test, y_pre_test)
        self.F1_test_total = f1_score(y_true_test, y_pre_test)
        # self.AUC_test_total = roc_auc_score(y_true_test, y_pre_test)
        return self.acc_test_total, self.pre_test_total, self.rec_test_total, self.F1_test_total, self.AUC_test_total


    def save(self, path): # 'D:/股价 小样本/原型网络/log2020 proto小样本/'
        torch.save(
            {'model': self.model,
             'output_dim': self.output_dim,
             'lr': self.lr,
             'epoch_size': self.epoch_size,
             'support_size': self.Ns,
             'query_size': self.Nq,
             'test_size': self.test_size},
            path + f'{self.acc_test_total:.2f} model_net_' + \
            str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())) + '.pkl')

    # 储存中心
    def save_center(self, path):
        datas = []
        for label in self.center.keys():
            datas.append([label] + list(self.center[label].detach().numpy()))
        with open(path, "w", newline="") as datacsv:
            csvwriter = csv.writer(datacsv, dialect=("excel"))
            csvwriter.writerows(datas)

    # 加载中心
    def load_center(self, path):
        csvReader = csv.reader(open(path))
        for line in csvReader:
            label = int(line[0])
            center = [float(line[i]) for i in range(1, len(line))]
            center = np.array(center)
            center = Variable(torch.from_numpy(center))
            self.center[label] = center