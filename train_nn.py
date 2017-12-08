import tensorflow as tf
import numpy as np
import math
import sys
import random
import os
import csv
import matplotlib
import matplotlib.pyplot as plt
import random

from model import Model

NUM_ENTRIES = 10000
NUM_TRAIN = 9000
NUM_VALIDATE = NUM_ENTRIES - NUM_TRAIN
BATCH_SIZE = 1
EPOCHS = [(100, 2e-3)]
RESTORE = True
PERFORM_TRAIN = False
TRAIN = 'training_data.txt'
logFile = open("log.txt", "w")

def get_dataset():
    trainX, trainY, testX, testY = [], [], [], []
    index = 0
    for line in open("training_data.txt"):
        split = line.split(", ")
        x = map(float, split[0].split(" "))
        trueLen = int(split[1])
        randomWalkLen = float(split[2])
        y = randomWalkLen

        if index < NUM_TRAIN:
            trainX.append(x)
            trainY.append(y)
        else:
            testX.append(x)
            testY.append(y)
        index += 1
    return trainX, trainY, testX, testY

def run_epoch(model, sess, X, Y, is_training, lr):
    if is_training:
        input_indices = np.arange(NUM_TRAIN)
    else:
        input_indices = np.arange(NUM_VALIDATE)
    np.random.shuffle(input_indices)
    num_batches = int((len(input_indices) + BATCH_SIZE - 1)/BATCH_SIZE)
    avg_loss = 0

    for b in range(num_batches):
        cur_indices = input_indices[(b*BATCH_SIZE):((b + 1)*BATCH_SIZE)]
        inputs_batch, labels_batch = [], []
        for idx in cur_indices:
            inputs_batch.append(X[idx])
            labels_batch.append(Y[idx])
        inputs_batch = np.array(inputs_batch)
        labels_batch = np.array(labels_batch)

        pred = None
        if is_training:
            loss, pred = model.train_on_batch(sess, inputs_batch, labels_batch, lr)
            avg_loss += loss/num_batches
        else:
            loss, pred = model.predict_on_batch(sess, inputs_batch, labels_batch)
            avg_loss += loss/num_batches
        
        #if not is_training:
        #    print "Predictions:"
        #    print pred
        #    print "True:"
        #    print labels_batch

        #if b % 100 == 0 and b > 0:
        #    print("completed %d batches" % b)

    if is_training:
        print "Training loss:", avg_loss
        logFile.write("Training loss: %.10f\n" % avg_loss)
    else:
        print "Validation loss:", avg_loss
        logFile.write("Validation loss: %.10f\n" % avg_loss)

    return avg_loss

def train(sess, model, saver):
    trainX, trainY, testX, testY = get_dataset()
    epoch = 0
    best_loss = 1E10
    for numEpochs, lr in EPOCHS:
        for i in range(numEpochs):
            print "Epoch", epoch
            cur_loss = run_epoch(model, sess, trainX, trainY, True, lr)
            cur_loss = run_epoch(model, sess, testX, testY, False, lr)
            if best_loss > cur_loss:
                print "better loss found, saving weights"
                saver.save(sess, 'model.weights')
                best_loss = cur_loss
            epoch += 1
    print "Best Loss:", best_loss

def plot_error(sess, model):
    trainX, trainY, testX, testY = get_dataset()
    
    allPred = None
    num_batches = int((len(testX) + BATCH_SIZE - 1)/BATCH_SIZE)
    for b in range(num_batches):
        x = testX[(b*BATCH_SIZE):((b + 1)*BATCH_SIZE)]
        y = testY[(b*BATCH_SIZE):((b + 1)*BATCH_SIZE)]
        loss, pred = model.predict_on_batch(sess, x, y)
        pred = np.abs(pred - y)
        if allPred is None:
            allPred = pred
        else:
            allPred = np.concatenate((allPred, pred), axis=0)
    
    yVal = [y for y in allPred]
    yVal.sort()
    plt.plot(np.arange(NUM_VALIDATE), yVal)
    plt.savefig('error_plot.png', format='png')

def main():
    model = None
    sess = None
    mean = None
    with tf.Graph().as_default():
        model = Model()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        tf.add_to_collection("pred", model.pred)

        if RESTORE:
            saver.restore(sess, 'model.weights')
        if PERFORM_TRAIN:
            train(sess, model, saver)
        else:
            plot_error(sess, model)

if __name__ == "__main__":
    main()
