import pandas as pd
import matplotlib.pyplot as plt
import random
import math
import sys
target_url = ("https://archive.ics.uci.edu/ml/machine-learning-"
"databases/undocumented/connectionist-bench/sonar/sonar.all-data")

#2.6 平行坐标图－ linePlots
#read rocks versus mines data into pandas data frame
rocksVMines = pd.read_csv(target_url,header=None)

for i in range(208):
    # assign color based on "M" or "R" labels
    if rocksVMines.iat[i, 60] == "M":
        pcolor = "red"
    else:
        pcolor = "blue"
    # plot rows of data as if they were series data
    dataRow = rocksVMines.iloc[i, 0:60]
    dataRow.plot(color=pcolor)

plt.xlabel("Attribute Index")
plt.ylabel(("Attribute Values"))
plt.title(" linePlots")
plt.show()

#2.7 属性对的交会图 -corrPlot
#calculate correlations between real-valued attributes
plt.subplot(1,2,1)
dataRow2 = rocksVMines.iloc[1,0:60]
dataRow3 = rocksVMines.iloc[2,0:60]
plt.scatter(dataRow2, dataRow3)
plt.xlabel("2nd Attribute")
plt.ylabel(("3rd Attribute"))
plt.title(" corrPlots")
plt.subplot(1,2,2)
dataRow21 = rocksVMines.iloc[20,0:60]
plt.scatter(dataRow2, dataRow21)
plt.xlabel("2nd Attribute")
plt.ylabel(("21st Attribute"))
plt.title(" corrPlots")
plt.show()

#2.8 分类问题标签和实数值属性之间的相关性 -targetCorr
#change the targets to numeric values
target = []
for i in range(208):
    # assign 0 or 1 target value based on "M" or "R" labels
    if rocksVMines.iat[i, 60] == "M":
        target.append(1.0)
    else:
        target.append(0.0)
#plot 35th attribute
dataRow = rocksVMines.iloc[0:208,35]
plt.scatter(dataRow, target)
plt.xlabel("Attribute Value")
plt.ylabel("Target Value")
plt.title(" targetCorr")
plt.show()

#To improve the visualization, this version dithers the points a little and makes them somewhat transparent
target = []
# assign 0 or 1 target value based on "M" or "R" labels
# and add some dither
for i in range(208):
    if rocksVMines.iat[i, 60] == "M":
        target.append(1.0 + random.uniform(-0.1, 0.1))
    else:
        target.append(0.0 + random.uniform(-0.1, 0.1))
dataRow = rocksVMines.iloc[0:208,35]
plt.scatter(dataRow, target, alpha=0.5, s=120)
plt.xlabel("Attribute Value")
plt.ylabel("Target Value")
plt.title(" targetCorr")
plt.show()

#2.9 皮尔森系数
#calculate correlations between real-valued attributes
dataRow2 = rocksVMines.iloc[1,0:60]
dataRow3 = rocksVMines.iloc[2,0:60]
dataRow21 = rocksVMines.iloc[20,0:60]
mean2 = 0.0; mean3 = 0.0; mean21 = 0.0
numElt = len(dataRow2)
for i in range(numElt):
    mean2 += dataRow2[i] / numElt
    mean3 += dataRow3[i] / numElt
    mean21 += dataRow21[i] / numElt
var2 = 0.0; var3 = 0.0; var21 = 0.0
for i in range(numElt):
    var2 += (dataRow2[i] - mean2) * (dataRow2[i] - mean2) / numElt
    var3 += (dataRow3[i] - mean3) * (dataRow3[i] - mean3) / numElt
    var21 += (dataRow21[i] - mean21) * (dataRow21[i] - mean21) / numElt
corr23 = 0.0; corr221 = 0.0
for i in range(numElt):
    corr23 += (dataRow2[i] - mean2) * \
              (dataRow3[i] - mean3) / (math.sqrt(var2 * var3) * numElt)
    corr221 += (dataRow2[i] - mean2) * \
               (dataRow21[i] - mean21) / (math.sqrt(var2 * var21) * numElt)
sys.stdout.write("Correlation between attribute 2 and 3 \n")
print(corr23)
sys.stdout.write(" \n")
sys.stdout.write("Correlation between attribute 2 and 21 \n")
print(corr221)
sys.stdout.write(" \n")

#2.10 用热图（heat map）展示属性和标签的相关性
#calculate correlations between real-valued attributes
corMat = pd.DataFrame(rocksVMines.corr())
plt.pcolor(corMat)
plt.show()

