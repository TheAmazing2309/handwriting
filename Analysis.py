import matplotlib.pyplot as plt
import numpy as np

x = []
y = []
yLoss = []
avyLoss = []

fig, (upper, lower) = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))

cutoff = 1152

with open("BatchMetrics.txt", "r") as f:
    avloss = 0
    for i, line in enumerate(f):
        epoch = int(line.split(",")[0].split("/")[-1]) - 1
        x.append(i+1)
        if int(line.split("batch: ")[1]) <= 500:
            y.append(int(line.split("batch: ")[1]))
        else:
            y.append(y[-1])
        yLoss.append(float(line.split(",")[2].split(" ")[-1]))
        avloss+=float(line.split(",")[2].split(" ")[-1])
        if i%cutoff == 0:
            avloss /= cutoff
            avyLoss.append(avloss)
            avloss=0

#upper.plot(x,y)
upper.plot(x,yLoss)
lower.plot([i+1 for i in range(len(avyLoss))], avyLoss)

xArr = np.array(x)
timeFit = np.polyfit(xArr, y, 1)
lossFit = np.polyfit(xArr, yLoss, 1)
#upper.plot(xArr, np.polyval(timeFit, xArr), '--', color='red')
upper.plot(xArr, np.polyval(lossFit, xArr), '--', color='red')


plt.show()