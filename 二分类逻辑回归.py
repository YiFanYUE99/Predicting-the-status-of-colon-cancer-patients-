# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 17:13:37 2022

@author: YiFan Yue
"""

import matplotlib.pyplot as plt
import numpy as np
#有更简单的导入方法了iloc函数
import pandas as pd
filename="D:\\作业\\210224结肠癌文章初稿\\LOG2\\positive.csv"
df=pd.read_csv(filename,header=0,index_col=0,encoding="gbk")#指定第一行为列名,第一列为行名

X=np.array(df.iloc[:,1:])


#手动给数据建立分类
y1=np.zeros((15,1))
y2=np.ones((57,1))
y=np.vstack((y1,y2))
y=y.ravel()#将数组拉成一维数组
y =np.array(list(map(int, y)))

#3.X是特征，y是分类；建立测试集训练集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =    train_test_split(X, y,
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)




#5用sklearn做逻辑回归
from sklearn.linear_model import LogisticRegression

#5.1关于C和正确率的关系
trainscore=[]
testscore=[]
Cindex=[]
for i in range(-2,12):
    lr = LogisticRegression(penalty='l1', C=10**i, solver='liblinear', multi_class='ovr')
    lr.fit(X_train, y_train)
    trainscore.append(lr.score(X_train, y_train))
    testscore.append(lr.score(X_test, y_test))
    Cindex.append(10**i)
trainscore=np.array(trainscore)
testscore=np.array(testscore)
Cindex=np.array(Cindex)
#画图，
plt.figure(figsize=(6, 6), dpi=100)#设置画布figsize大小、dpi像素
plt.plot(Cindex, trainscore,marker='o',color='green',label='train accuracy')
plt.plot(Cindex,testscore,marker='o',color='blue',label='test accuracy')
plt.legend(loc='lower right')
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.xlim([10**-2,10**12])
plt.xscale('log')#使得横坐标log缩放显示
plt.title('Regularization parameters-Accuracy')
plt.show()
#得出结论C=100时正确率最高



#5.2 L1正则化调整参数，不要用SBS
#拟合模型
lr = LogisticRegression(penalty='l1', C=100, solver='liblinear', multi_class='ovr')
lr.fit(X_train, y_train)
#查看模型正确率
print('Training accuracy:', lr.score(X_train, y_train))
print('Test accuracy:', lr.score(X_test, y_test))

#6.看各个特征的权重
weights=pd.DataFrame(lr.coef_)
weights.columns=df.columns[1:]
weights=weights.T
weights.to_csv("D:\\作业\\210224结肠癌文章初稿\\LOG2\\binary\\逻辑回归\\权重.csv",header=True,index=True);



#7.计算某个样本属于某类的概率(也可以用于预测)
lr.predict_proba(X_test[:3, :])#前三行属于某类的概率
lr.predict_proba(X_test[:3, :]).argmax(axis=1)#识别每行最大列的值得到预测的分类标签
lr.predict(X_test[:3, :])#直接返回标签的另外一种方法
#将某行数据转换为二维数组
X_test[0, :].reshape(1, -1)#应该是1行-1列的数据，实际是只有一行一列的数据
#预测某个样本的分类标签
lr.predict(X_test[0, :].reshape(1, -1))
y_pred=lr.predict_proba(X_test)
y_pred1=y_pred[:,1]



#8.交叉验证
from sklearn.model_selection import cross_val_score
scores=cross_val_score(estimator=lr,
                       X=X_train,
                       y=y_train,
                       cv=10,
                       n_jobs=-1)
print('CV accuracy:%.3f+/-%.3f'%(np.mean(scores),np.std(scores)))


#8.画混淆矩阵
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
#画混淆矩阵
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)#这是一个把矩阵或者数组绘制成图像的函数
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()






#9.画ROC曲线

from sklearn.metrics import roc_curve,auc
fpr,tpr,th=roc_curve(y_test, y_pred,pos_label=1)
for i,value in enumerate(th):
    print("%f %f %f"%(fpr[i],tpr[i],value))
roc_auc=auc(fpr,tpr)
plt.plot(fpr,tpr,'k--',label='ROC(area={0:.2f})'.format(roc_auc),lw=2)
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


