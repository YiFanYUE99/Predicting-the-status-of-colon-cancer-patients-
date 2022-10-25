# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:18:49 2021

@author: YiFan Yue
"""

import matplotlib.pyplot as plt
import numpy as np
#有更简单的导入方法了iloc函数
import pandas as pd
filename="D:\\作业\\210224结肠癌文章初稿\\LOG2\\positive.csv"
df=pd.read_csv(filename,header=0,index_col=0,encoding="gbk")#指定第一行为列名,第一列为行名
#1.设置特征和分类
X=np.array(df.iloc[:,1:])

y=np.array(df.iloc[:,0])

#2.给变量加标签以便模型评估，不要用独热编码，逻辑回归无法识别数组类别；手动调整特征
#from sklearn.preprocessing import OneHotEncoder
#sickness=OneHotEncoder()
#y=sickness.fit_transform(y.reshape(-1,1)).toarray()
sickness={'V':0,
          'C1':1,
          'C2':2,
          'C3':3,
          'C4':4}
y=pd.Series(y).map(sickness)#注意，只有Series格式能用.map映射特征

#3.X是特征，y是分类；建立测试集训练集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =    train_test_split(X, y,
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)


#构建决策树
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, #最优超参数为4
                                    random_state=1)
tree_model.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
#调整决策树的超参数
from sklearn.model_selection import GridSearchCV
param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}]
gs = GridSearchCV(estimator=tree_model, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,#在搜索参数结束后，用最佳参数结果再次fit一遍全部数据集
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(X_train, y_train)
print(gs.best_score_)#获得性能最优模型的准确率：0.8
print(gs.best_params_)#最优树深度为4
#用独立的测试数据集评估最优模型的性能
clf = gs.best_estimator_#因为设置了refit=TRUE,不需要clf.fit(X_train, y_train) 
print('Test accuracy: %.3f' % clf.score(X_test, y_test))



#画决策树
filename2="D:\\作业\\210224结肠癌文章初稿\\LOG2\\metabolites.csv"
me=pd.read_csv(filename2)#指定第一行为列名,第一列为行名
from sklearn import tree
import matplotlib.pyplot as plt
cn=['healthy','C1','C2','C3','C4']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (30,20), dpi=60)
tree.plot_tree(tree_model,
               feature_names = me.columns, 
               class_names=cn,
               filled = True)
plt.tight_layout()



from sklearn.model_selection import cross_val_score
#分层k折交叉验证法简洁得评估模型
scores = cross_val_score(estimator=tree_model,
                         X=X_train,
                         y=y_train,
                         cv=10,#十折交叉验证
                         n_jobs=16)
print('CV accuracy scores: %s' % scores)#输出每次的交叉验证准确率
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))#准确率的均值和方差


#混淆矩阵
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#计算测试集的混淆矩阵
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

#画测试集的混淆矩阵
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)#这是一个把矩阵或者数组绘制成图像的函数
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()




#计算训练集的混淆矩阵
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_train)
confmat = confusion_matrix(y_true=y_train, y_pred=y_pred)

#画训练集的混淆矩阵
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)#这是一个把矩阵或者数组绘制成图像的函数
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()













