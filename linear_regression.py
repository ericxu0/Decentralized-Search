#!/usr/bin/env python

import matplotlib.pyplot as plot
import random
from sklearn.linear_model import LinearRegression

TEST_AMOUNT = 0.2

trainX = []
trainY = []
testX = []
testY = []
for line in open("training_data.txt"):
    split = line.split(" , ")
    x = map(float, split[0].split(" "))
    y = float(split[1])
    if random.random() < TEST_AMOUNT:
        testX.append(x)
        testY.append(y)
    else:
        trainX.append(x)
        trainY.append(y)

linreg = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
linreg.fit(trainX, trainY)

predictedY = linreg.predict(testX)
plot.scatter(testY, predictedY)
plot.title("Actual vs. Predicted Path Lengths")
plot.xlabel("Actual path length")
plot.ylabel("Predicted path length")
plot.show()

print "Linear regression"
print "Score: " + str(linreg.score(testX, testY))
print "Coef: " + str(linreg.coef_)
