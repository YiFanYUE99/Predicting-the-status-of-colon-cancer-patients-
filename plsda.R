tes1<-read.csv("D:/作业/210224结肠癌文章初稿/原始数据，峰面积/VIP/datamatrix.csv",header = T,row.names = 1)
tes2<-read.csv("D:/作业/210224结肠癌文章初稿/原始数据，峰面积/VIP/samplemetadata01.csv",header = T,row.names = 1)
ClassFc <-tes2$Class
ClassFc<-factor(ClassFc)
#PLSDA
library(ropls)
tes.plsda <- opls(tes1, ClassFc, orthoI = 0)
#画图
plot(tes.plsda,
     typeVc="x-score",
     parAsColFcVn=ClassFc,
     parLabVc=as.character(sampleMetaData[,"Class"]),
     parPaletteVc=c("pink","grey","green","blue","red")
     )
#求VIP值大于1的东西
vipVn <- tes.plsda@vipVn  # getVipVn()
vipVn
vipVn_select <- vipVn[vipVn > 1] 
#输出为表格
write.table(vipVn_select,"D:/作业/210224结肠癌文章初稿/原始数据，峰面积/VIP\\PLSDA-VIP01.csv",row.names=T,col.names=TRUE,sep=",")#再手动降序处理
#执行OPLS-DA
tes.oplsda <- opls(tes1, ClassFc, orthoI = NA)
#画图
plot(tes.oplsda,
     typeVc="x-score",
     parAsColFcVn=ClassFc,
     parLabVc=as.character(sampleMetaData[,"Class"]),
     parPaletteVc=c("pink","grey","green","blue","red")
)
#求VIP同上
vipVn1 <- tes.oplsda@vipVn  # getVipVn()
vipVn1
vipVn1_select <- vipVn1[vipVn1 > 1]
write.table(vipVn1_select,"OPLSDA-VIP.csv",row.names=T,col.names=TRUE,sep=",")