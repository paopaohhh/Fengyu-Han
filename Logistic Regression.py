#-*- coding : utf-8 -*-
# coding: utf-8

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import linear_model, datasets
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df_2020 = pd.read_csv('./data/Key Indicators/stock2020 Key Indicators.csv.csv')
# df_2020 = pd.read_csv('./data/Expanded Indicators/stock2020 Expanded Indicators.csv.csv')
column_list = df_2020.columns.to_list()
column_list.remove('label')
X_data = df_2020[column_list]
y_data = df_2020['label']
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=0)

scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score), "Precision": make_scorer(precision_score), "Recall": make_scorer(recall_score), "F1": make_scorer(f1_score)}
parameters = {'penalty': ('l1', 'l2'),'C': (0.01, 0.1,0.2,0.3, 0.5, 0.6,0.7,1)}

# 训练
gs = GridSearchCV(
    LogisticRegression(),
    param_grid=parameters,
    scoring=scoring,
    refit="AUC",#"AUC"是scoring字典里第1项的key
    return_train_score=True,
    cv=5,
)

gs.fit(X_train, y_train) #fit，拟合，即训练
results = gs.cv_results_
best_index = gs.best_index_

print('训练平均准确率：%.3f, 验证平均准确率：%.3f；'% (results['mean_train_Accuracy'][best_index], results['mean_test_Accuracy'][best_index]))
print('训练平均AUC：%.3f, 验证平均AUC：%.3f；'% (results['mean_train_AUC'][best_index], results['mean_test_AUC'][best_index]))
print('训练平均精度：%.3f, 验证平均精度：%.3f；'% (results['mean_train_Precision'][best_index], results['mean_test_Precision'][best_index]))
print('训练平均召回率：%.3f, 验证平均召回率：%.3f；'% (results['mean_train_Recall'][best_index], results['mean_test_Recall'][best_index]))
print('训练平均F1：%.3f, 验证平均F1：%.3f。' % (results['mean_train_F1'][best_index], results['mean_test_F1'][best_index]))

best_model = gs.best_estimator_

y_predict = best_model.predict(X_test) #在单分出的测试集上测

print('在测试集上，准确率：%.4f， AUC：%.4f，精度：%.4f，召回率：%.4f，F1：%.4f。' % (accuracy_score(y_test, y_predict), roc_auc_score(y_test, y_predict), precision_score(y_test, y_predict), recall_score(y_test, y_predict), f1_score(y_test, y_predict)))

df_result = pd.DataFrame([['Logistic Regression', results['mean_train_Accuracy'][best_index],results['mean_train_AUC'][best_index],results['mean_train_Precision'][best_index],results['mean_train_Recall'][best_index],results['mean_train_F1'][best_index],\
                                            results['mean_test_Accuracy'][best_index],results['mean_test_AUC'][best_index],results['mean_test_Precision'][best_index],results['mean_test_Recall'][best_index],results['mean_test_F1'][best_index],\
                         accuracy_score(y_test, y_predict), roc_auc_score(y_test, y_predict), precision_score(y_test, y_predict), recall_score(y_test, y_predict), f1_score(y_test, y_predict)]],\
                         columns=['模型','训练Accuracy', '训练AUC','训练Precision','训练Recall', '训练F1',\
                                            '验证Accuracy', '验证AUC','验证Precision','验证Recall', '验证F1', '测试Accuracy', '测试AUC','测试Precision','测试Recall', '测试F1'])

df_result.to_csv('./测试结果.csv',mode='a', index=0,header=None)
# df_result.to_csv('./筛选数据测试结果.csv',mode='a', index=0,header=None)