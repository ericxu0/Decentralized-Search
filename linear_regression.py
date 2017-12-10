#!/usr/bin/env python

import random
from collections import Counter
from sklearn.linear_model import Ridge
from os import listdir
from os.path import isfile, join

#TEST_AMOUNT = 0.2

fileDir = "data/training_data/"
files = [f for f in listdir(fileDir) if isfile(join(fileDir, f)) and f.startswith("training_data_") and f.endswith(".txt")]

allX_sim = []
allY_sim = []
allX_n2v = []
allY_n2v = []

for filePath in files:
    n2v = filePath.endswith("_n2v.txt")
    random.seed(42)
    trainX = []
    trainY = []

    for line in open(fileDir + filePath):
        split = line.split(", ")

        x = map(float, split[0].split(" "))

        trueLen = int(split[1])
        randomWalkLen = float(split[2]) # Not used, trueLen gives very slightly better results
        y = trueLen

        if trueLen <= 1:
            continue # Meaningless to train if we're at the destination or 1 edge away

        trainX.append(x)
        trainY.append(y)
        if n2v:
            allX_n2v.append(x)
            allY_n2v.append(y)
        else:
            allX_sim.append(x)
            allY_sim.append(y)

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

reg = Ridge(fit_intercept=False)
reg.fit(allX_n2v, allY_n2v)
outputPath = fileDir + "Ridge" + "_" + "all_n2v" + ".weights"
outputFile = open(outputPath, "w")
outputFile.write(str(reg.coef_)[1:-1])
outputFile.close()
print "Wrote to path:", outputPath

reg = Ridge(fit_intercept=False)
reg.fit(allX_sim, allY_sim)
outputPath = fileDir + "Ridge" + "_" + "all_sim" + ".weights"
outputFile = open(outputPath, "w")
outputFile.write(str(reg.coef_)[1:-1])
outputFile.close()
print "Wrote to path:", outputPath