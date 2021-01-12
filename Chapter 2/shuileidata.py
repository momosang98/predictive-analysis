import urllib.request
import sys
import pylab
import numpy as np
import pandas as pd
import scipy.stats as stats

#2.1 read data from uci data repository

target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
data = urllib.request.urlopen(target_url)
#arrange data into list for labels and list of lists for attributes
xList = []
labels = []
for line in data:
    # split on comma
    row = line.decode().strip().split(",")
    xList.append(row)
sys.stdout.write("Number of Rows of Data = " + str(len(xList)) + '\n')
sys.stdout.write("Number of Columns of Data = " + str(len(xList[1]))+ '\n')


#2.2 check data_type

nrow = len(xList)
ncol = len(xList[1])
type = [0]*3
colCounts = []
for col in range(ncol):
    for row in xList:
        try:
            a = float(row[col])
            if isinstance(a, ﬂoat):
                type[0] += 1
        except ValueError:
            if len(row[col]) > 0:
                type[1] += 1
            else:
                type[2] += 1
    colCounts.append(type)
    type = [0] * 3
sys.stdout.write("Col#" + '\t' + "Number" + '\t' +
"Strings" + '\t ' + "Other\n")
iCol = 0
for types in colCounts:
    sys.stdout.write(str(iCol) + '\t\t' + str(types[0]) + '\t\t' +
                     str(types[1]) + '\t\t' + str(types[2]) + "\n")
    iCol += 1

    
#2.3 Summary

#arrange data into list for labels and list of lists for attributes
labels = []
#generate summary statistics for column 3 (e.g.)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))
colArray = np.array(colData)
colMean = np.mean(colArray)
colsd = np.std(colArray)
sys.stdout.write("Mean = " + '\t' + str(colMean) + '\t\t' +
                 "Standard Deviation = " + '\t ' + str(colsd) + "\n")

#calculate quantile boundaries
ntiles = 4
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i * (100) / ntiles))
sys.stdout.write("\nBoundaries for 4 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

#run again with 10 equal intervals
ntiles = 10
percentBdry = []
for i in range(ntiles+1):
    percentBdry.append(np.percentile(colArray, i * (100) / ntiles))
sys.stdout.write("Boundaries for 10 Equal Percentiles \n")
print(percentBdry)
sys.stdout.write(" \n")

#The last column contains categorical variables
col = 60
colData = []
for row in xList:
    colData.append(row[col])
unique = set(colData)
sys.stdout.write("Unique Label Values \n")
print(unique)

#count up the number of elements having each value
catDict = dict(zip(list(unique),range(len(unique))))
catCount = [0]*2
for elt in colData:
    catCount[catDict[elt]] += 1
sys.stdout.write("\nCounts for Each Value of Categorical Label \n")
print(list(unique))
print(catCount)

'''so, the last number 4 is outlier'''


#2.4 qqplotAttribute 用分位数图展示异常点

#generate summary statistics for column 3 (e.g.)
col = 3
colData = []
for row in xList:
    colData.append(float(row[col]))
stats.probplot(colData, dist="norm", plot= pylab)
pylab.show()


#2.5 用Python Pandas 读入数据、分析数据
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")
#read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(target_url,header=None)

#print head and tail of data frame, 前5后5行
print(rocksVMines.head())
print(rocksVMines.tail())

#print summary of data frame
summary = rocksVMines.describe()
print(summary)
