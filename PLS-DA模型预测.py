# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 20:31:20 2022

@author: yj991
"""
from sklearn.cross_decomposition import PLSRegression
model = PLSRegression()

import numpy as np 
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

import pandas  as pd
import matplotlib.pyplot as plt



filename="D:\\作业\\210224结肠癌文章初稿\\LOG2\\positive.csv"
df=pd.read_csv(filename,header=0,index_col=0,encoding="gbk")#指定第一行为列名,第一列为行名

X=np.array(df.iloc[:,1:])


#手动给数据建立分类
y1=np.zeros((15,1))
y2=np.ones((14,1))
y3=2*np.ones((13,1))
y4=3*np.ones((16,1))
y5=4*np.ones((14,1))
y=np.vstack((y1,y2,y3,y4,y5))
y=y.ravel()#将数组拉成一维数组
y =np.array(list(map(int, y)))

#区分训练集测试集
X_train, X_test, y_train, y_test =    train_test_split(X, y,
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

y_train=pd.get_dummies(y_train)#独热编码，只要训练集转化为独热编码



#建模
model = PLSRegression(n_components=12)#超参数调整为12
model.fit(X_train,y_train)


#预测
y_pred = model.predict(X_test)

#将预测结果（类别矩阵）转换为数值标签
y_pred = np.array([np.argmax(i) for i in y_pred])

#交叉验证
from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=model,
                       X=X_train,
                       y=y_train,
                       cv=10,
                       n_jobs=-1)
print('CV accuracy:%.3f+/-%.3f'%(np.mean(scores),np.std(scores)))

#十折交叉验证准确率CV accuracy:0.209+/-0.236

#模型评价
from sklearn.metrics import confusion_matrix
print('测试集混淆矩阵为：\n',confusion_matrix(y_test,y_pred))
confmat=confusion_matrix(y_true=y_test, y_pred=y_pred)




# 绘制混淆矩阵


fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)#这是一个把矩阵或者数组绘制成图像的函数
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()























