# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:06:00 2022

@author: yj991
"""

import numpy as np
#有更简单的导入方法了iloc函数
import pandas as pd
filename1="D:\\作业\\210224结肠癌文章初稿\\LOG2\\统计分析\\t检验\\独立样本的双侧t检验结果.csv"
df=np.array(pd.read_csv(filename1,header=0,index_col=None,encoding="gbk"))

filename2="D:\\作业\\210224结肠癌文章初稿\\LOG2\\PCA+PLS-DA\\PLSDA-VIP01.csv"
df2=np.array(pd.read_csv(filename2,header=0,index_col=None,encoding="gbk"))


select=[]
df2.shape[0]#240
for i in range(df.shape[0]):
    for j in range(df2.shape[0]):
        if df[i,0]==df2[j,0]:
            select.append(df[i,0])

selected=pd.DataFrame(select)

#上述过程挑选出VIP大于1，p<0.05的代谢物


select1=[]
filename3="D:\\作业\\210224结肠癌文章初稿\\LOG2\\metabolites.csv"
df3=np.array(np.array(pd.read_csv(filename3,header=None,index_col=None)))

selected2=np.array(selected)


for i in range(selected2.shape[0]):
    for j in range(df3.shape[1]):
        if selected2[i,0]==df3[1,j]:
            select1.append(df3[:,j])
            
selected1=pd.DataFrame(select1)

selected1.to_csv("D:\\作业\\210224结肠癌文章初稿\\LOG2\\统计分析\\metaselect.csv",header=False,index=False);

#注释代谢物