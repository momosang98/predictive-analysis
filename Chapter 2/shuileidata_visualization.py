import pandas as pd
import matplotlib.pyplot as plt
import random
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

#2.9