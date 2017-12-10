#!/usr/bin/env python

import random
from collections import Counter
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn.preprocessing import PolynomialFeatures
from os import listdir
from os.path import isfile, join

#TEST_AMOUNT = 0.2

USE_POLY = False

if USE_POLY:
    poly = PolynomialFeatures(degree=2)

trainX = []
trainY = []

fileDir = "data/training_data/"
files = [f for f in listdir(fileDir) if isfile(join(fileDir, f)) and f.startswith("training_data_") and f.endswith(".txt")]

for filePath in files:
    random.seed(42)

    for line in open(fileDir + filePath):
        split = line.split(", ")

        x = map(float, split[0].split(" "))

        trueLen = int(split[1])
        randomWalkLen = float(split[2])
        y = randomWalkLen # TODO: trueLen?

        if trueLen <= 1:
            continue # Meaningless to train if we're at the destination or 1 edge away

        trainX.append(x)
        trainY.append(y)

    if USE_POLY:
        trainX = poly.fit_transform(trainX)

    reg = Ridge(fit_intercept=False)
    reg.fit(trainX, trainY)

    name = filePath[len("training_data_") : filePath.rfind('.')]
    typeStr = str(type(reg))
    typeStr = typeStr[typeStr.rfind('.') + 1 : typeStr.rfind('\'')]
    outputPath = fileDir + typeStr + "_" + name + ".weights"
    outputFile = open(outputPath, "w")
    outputFile.write(str(reg.coef_)[1:-1])
    outputFile.close()
    print "Wrote to path:", outputPath
