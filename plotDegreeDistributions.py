import snap
import numpy as np
import matplotlib.pyplot as plt


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    
    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    X, Y = [], []

    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)
    numberTotalNodes = Graph.GetNodes()
    for item in DegToCntV:
        numNodes = item.GetVal2()
        degreeNode = item.GetVal1()
        if degreeNode == 0:
            continue
        X.append(degreeNode)
        Y.append(numNodes / float(numberTotalNodes))


    ############################################################################
    return X, Y


    

def plotGraphs():
    
    facebookNet = snap.LoadEdgeList(snap.PUNGraph, "data/real/facebook_combined.txt", 0, 1)
    collabNet = snap.LoadEdgeList(snap.PUNGraph, "data/real/ca-HepTh.txt", 0, 1)
    citationNet = snap.LoadEdgeList(snap.PUNGraph, "data/real/cit-HepTh.txt", 0, 1)

    erdosRenyi = snap.LoadEdgeList(snap.PUNGraph, "data/synthetic/gnm0.txt", 0, 1)
    smallWorld = snap.LoadEdgeList(snap.PUNGraph, "data/synthetic/smallworld0.txt", 0, 1)
    prefAttachment = snap.LoadEdgeList(snap.PUNGraph, "data/synthetic/prefattach0.txt", 0, 1)

    x_facebook, y_facebook = getDataPointsToPlot(facebookNet)
    plt.loglog(x_facebook, y_facebook, color = 'y', label = 'Facebook Network')

    x_collab, y_collab = getDataPointsToPlot(collabNet)
    plt.loglog(x_collab, y_collab, color = 'g', label = 'Collaboration Network')

    x_citation, y_citation = getDataPointsToPlot(citationNet)
    plt.loglog(x_citation, y_citation, color = 'b', label = 'Citation Network')

    x_erdos, y_erdos = getDataPointsToPlot(erdosRenyi)
    plt.loglog(x_erdos, y_erdos, color = 'r', label = 'Erdos Renyi Network')

    x_smallworld, y_smallworld = getDataPointsToPlot(smallWorld)
    plt.loglog(x_smallworld, y_smallworld, color = 'c', label = 'Small World Network')

    x_prefAttachment, y_prefAttachment = getDataPointsToPlot(prefAttachment)
    plt.loglog(x_prefAttachment, y_prefAttachment, color = 'k', label = 'Preferential Attachment Network')
    

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution')
    plt.legend()
    plt.show()

plotGraphs()