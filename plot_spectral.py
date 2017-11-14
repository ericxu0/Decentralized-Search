import snap
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

INPUT_DIR = "data/synthetic/"
OUTPUT_DIR = "plots/"

def plot_spectral(filename, index):
    Graph = snap.LoadEdgeList(snap.PUNGraph, INPUT_DIR + filename + '.txt', 0, 1)
    v2 = []
    v3 = []
    f = open(INPUT_DIR + filename + '.out', "r")
    first = True
    N = -1
    for line in f:
        e = line.split(' ')
        if first:
            first = False
            N = int(e[0])
            continue
        v2.append(float(e[1]))
        v3.append(float(e[2]))
    
    plt.figure(index)
    plt.scatter(v2, v3)
    
    for x in range(N):
        for y in range(x + 1, N):
            if Graph.IsEdge(x, y):
                plt.plot([v2[x], v2[y]], [v3[x], v3[y]])
    
    plt.xlabel('Second Smallest Eigenvector')
    plt.ylabel('Third Smallest Eigenvector')
    plt.savefig(OUTPUT_DIR + filename + '.png', format='png')


plot_spectral("gnm_small0", 0);
plot_spectral("smallworld_small0", 1);
plot_spectral("powerlaw_small0", 2);
plot_spectral("prefattach_small0", 3);
