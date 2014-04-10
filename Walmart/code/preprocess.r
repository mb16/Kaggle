library(zoo)
setwd("C:/Users/mbrandt/Documents/kaggle/walmart/code")

dfStore <- read.csv(file='../data/stores.csv')
dfTrain <- read.csv(file='../data/train.csv')
dfTest <- read.csv(file='../data/test.csv')
dfFeatures <- read.csv(file='../data/features.csv')

# Merge Type and Size
dfTrainTmp <- merge(x=dfTrain, y=dfStore, all.x=TRUE)
dfTestTmp <- merge(x=dfTest, y=dfStore, all.x=TRUE)

# Merge all the features
dfTrainMerged <- merge(x=dfTrainTmp, y=dfFeatures, all.x=TRUE)
dfTestMerged <- merge(x=dfTestTmp, y=dfFeatures, all.x=TRUE)

dfTrainMerged <- dfTrainMerged[order(dfTrainMerged$Store, dfTrainMerged$Dept, dfTrainMerged$Date), ]
dfTestMerged <- dfTestMerged[order(dfTestMerged$Store, dfTestMerged$Dept, dfTestMerged$Date), ]

# ---------------------------------------------


# Save datasets
write.table(x=dfTrainMerged$Weekly_Sales,
            file='trainTarget.csv',
            sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)

quit()

# ---------------------------------------------

createID <- function(store, dept, date)
{
  return (paste(as.character(store), as.character(dept), as.character(date), sep="_"))  
}

trainIDs <- mapply(createID, dfTrainMerged$Store, dfTrainMerged$Dept, dfTrainMerged$Date)
testIDs <- mapply(createID, dfTestMerged$Store, dfTestMerged$Dept, dfTestMerged$Date)


# Save datasets
write.table(x=trainIDs,
            file='trainIDs.csv',
            sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)
write.table(x=testIDs,
            file='testIDs.csv',
            sep=',', row.names=FALSE, col.names=FALSE, quote=FALSE)


# ---------------------------------------------




weightFunc <- function(hol)
{
  if (hol == TRUE) return(5);
  return (1);
}


dfTrainMerged$Weight <- mapply(weightFunc, dfTrainMerged$IsHoliday)
dfTestMerged$Weight <- mapply(weightFunc, dfTestMerged$IsHoliday)




setHoliday <- function(dt,hol)
{
  #print (as.Date(as.character(dt), "%Y-%m-%d"))
  mon <- as.numeric(format(as.Date(as.character(dt), "%Y-%m-%d"), "%m"))

  if ( hol == FALSE) {return (0)}
  return (mon)
  
}

dfTrainMerged$IsHoliday <- mapply(setHoliday, dfTrainMerged$Date, dfTrainMerged$IsHoliday)
dfTestMerged$IsHoliday <- mapply(setHoliday, dfTestMerged$Date, dfTestMerged$IsHoliday)



weekNum <- function(dt)
{
  as.numeric( format(as.Date(as.character(dt), "%Y-%m-%d"), "%U" ))  
}


dfTrainMerged$Date <- mapply(weekNum, dfTrainMerged$Date)
dfTestMerged$Date <- mapply(weekNum, dfTestMerged$Date)


typeToNum <- function(type)
{
  if (type == "A") return(1);
  if ( type == "B") return(2);
  return (3);
}


dfTrainMerged$Type <- mapply(typeToNum, dfTrainMerged$Type)
dfTestMerged$Type <- mapply(typeToNum, dfTestMerged$Type)


write.table(x=dfTrainMerged,
            file='trainAll.csv',
            sep=',', row.names=FALSE, quote=FALSE)
write.table(x=dfTestMerged,
            file='testAll.csv',
            sep=',', row.names=FALSE, quote=FALSE)


dfTrainMerged <- dfTrainMerged[!(is.na(dfTrainMerged$MarkDown1) & is.na(dfTrainMerged$MarkDown2) & is.na(dfTrainMerged$MarkDown3) & is.na(dfTrainMerged$MarkDown4) & is.na(dfTrainMerged$MarkDown5)),]
dfTestMerged <- dfTestMerged[!(is.na(dfTestMerged$MarkDown1) & is.na(dfTestMerged$MarkDown2) & is.na(dfTestMerged$MarkDown3) & is.na(dfTestMerged$MarkDown4) & is.na(dfTestMerged$MarkDown5)),]


dfTrainMerged$MarkDown1 <- na.approx(dfTrainMerged$MarkDown1)
dfTrainMerged$MarkDown2 <- na.approx(dfTrainMerged$MarkDown2)
dfTrainMerged$MarkDown3 <- na.approx(dfTrainMerged$MarkDown3)
dfTrainMerged$MarkDown4 <- na.approx(dfTrainMerged$MarkDown4)
dfTrainMerged$MarkDown5 <- na.approx(dfTrainMerged$MarkDown5)

dfTestMerged$MarkDown1 <- na.approx(dfTestMerged$MarkDown1)
dfTestMerged$MarkDown2 <- na.approx(dfTestMerged$MarkDown2)
dfTestMerged$MarkDown3 <- na.approx(dfTestMerged$MarkDown3)
dfTestMerged$MarkDown4 <- na.approx(dfTestMerged$MarkDown4)
dfTestMerged$MarkDown5 <- na.approx(dfTestMerged$MarkDown5)

# does not apply to train.
dfTestMerged$Unemployment <- na.locf(dfTestMerged$Unemployment) #  fills in last value rather than interpolate
dfTestMerged$CPI <- na.locf(dfTestMerged$CPI)




# Save datasets
write.table(x=dfTrainMerged,
file='train.csv',
sep=',', row.names=FALSE, quote=FALSE)
write.table(x=dfTestMerged,
file='test.csv',
sep=',', row.names=FALSE, quote=FALSE)

