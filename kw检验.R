library(lmPerm)
crcdata<-read.csv("D:/作业/210224结肠癌文章初稿/LOG2/positive.csv",header = TRUE,stringsAsFactors = F)
Pvaluekwcrc<-c(rep(0,ncol(crcdata)-2))
for(i in 3:ncol(crcdata))
{
  y1<-kruskal.test(crcdata[,i]~Class,data=crcdata)
  Pvaluekwcrc[i-2]<-y1[["p.value"]]
}
Pvaluekwcrc<-as.data.frame(Pvaluekwcrc)

write.table(Pvaluekwcrc,"D:/作业/210224结肠癌文章初稿/LOG2/统计分析/kruskal-wallis检验结果.csv",row.names=TRUE,col.names=FALSE,sep=",")

