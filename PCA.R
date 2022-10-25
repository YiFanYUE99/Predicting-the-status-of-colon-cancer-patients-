library(ropls)
dataMatrix<-read.csv("D:/作业/210224结肠癌文章初稿/原始数据，峰面积/VIP/datamatrix.csv",header = TRUE,row.names = 1,stringsAsFactors = F)#行名是样品，列名是代谢物
sacurine.pca<-opls(dataMatrix)
sampleMetaData<-read.csv("D:/作业/210224结肠癌文章初稿/原始数据，峰面积/VIP/samplemetadata.csv",header = TRUE,row.names = 1,stringsAsFactors = F)
ClassFc <- sampleMetaData[, "Class"]
plot(sacurine.pca,
     typeVc = "x-score",
     parAsColFcVn = ClassFc,#按照gender分类
     #parLabVc=as.character(sampleMetaData[,"Class"]),#点上显示分组，这行删掉什么都不显示
     parPaletteVc=c("blue","red","green","light blue","violet"))#指定每组的颜色
