#!/usr/bin/env python

import matplotlib.pyplot as plot
import random
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

TEST_AMOUNT = 0.2

USE_POLY = False

if USE_POLY:
    poly = PolynomialFeatures(degree=2)

trainX = []
trainY = []
testX = []
testY = []

for line in open("training_data_100k.txt"):
    split = line.split(", ")

    x = map(float, split[0].split(" "))
    x = x[2:] # TODO: change this if data changes!!

    trueLen = int(split[1])
    randomWalkLen = float(split[2])
    y = randomWalkLen # TODO: trueLen?

    if trueLen <= 1:
        continue # Meaningless to train if we're at the destination or 1 edge away

    if random.random() < TEST_AMOUNT:
        testX.append(x)
        testY.append(y)
    else:
        trainX.append(x)
        trainY.append(y)

if USE_POLY:
    trainX = poly.fit_transform(trainX)
    testX = poly.fit_transform(testX)

linreg = LinearRegression(fit_intercept=False, normalize=False, copy_X=True, n_jobs=1)
linreg.fit(trainX, trainY)
predictedY = linreg.predict(testX)

def sizes(x, y):
    c = Counter((x[i], y[i]) for i in range(len(x)))
    return [c[(x[i], y[i])] ** 2 * 5 for i in range(len(x))]

print "Linear regression"
print "Score: " + str(linreg.score(testX, testY))
print "Coef: " + str(linreg.coef_)

plot.scatter(testY, predictedY, s=sizes(testY, predictedY), marker='.', facecolor='none', edgecolor='b')
plot.title("Actual vs. Predicted Path Lengths")
plot.xlabel("Actual path length")
plot.ylabel("Predicted path length")
plot.show()
