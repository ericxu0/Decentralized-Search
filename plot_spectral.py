import snap
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

def plot_spectral(filename, index):
    Graph = snap.LoadEdgeList(snap.PUNGraph, filename[:-4] + '.txt', 0, 1)
    v2 = []
    v3 = []
    f = open(filename, "r")
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
    plt.savefig(filename[:-4] + '.png', format='png')


plot_spectral("data/synthetic/gnm_small0.out", 0);
plot_spectral("data/synthetic/smallworld_small0.out", 1);
plot_spectral("data/synthetic/prefattach_small0.out", 2);
