library(MASS)
crcdata<-read.csv("D:/作业/210224结肠癌文章初稿/LOG2/统计分析/t检验/positive.csv",header = TRUE,stringsAsFactors = F)
Pvaluecrc<-c(rep(0,ncol(crcdata)-2))
for(i in 3:ncol(crcdata))
{
  y1<-t.test(crcdata[,i]~crcdata[,2],data=crcdata)
  Pvaluecrc[i-2]<-y1[["p.value"]]
}
Pvaluecrc<-as.data.frame(Pvaluecrc)

write.table(Pvaluecrc,"D:/作业/210224结肠癌文章初稿/LOG2/统计分析/t检验/独立样本的双侧t检验结果.csv",row.names=TRUE,col.names=FALSE,sep=",")
