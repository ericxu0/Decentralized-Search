#include "Snap.h"
#include "utils.cpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <set>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

double predictPathLength(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    vector<double> feat;
    getFeatureVector(G, cur, dst, visited, feat);

    assert(feat.size() == ridgeWeights.size());
    double dot = 0.0;
    for (size_t i = 0; i < feat.size(); i++)
        dot += feat[i] * ridgeWeights[i];
    return dot;
}

int ridgeStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = 1E20;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = predictPathLength(G, nxt, dst, visited);
        if (nxtVal < bestVal) {
            bestVal = nxtVal;
            choices.clear();
        }
        if (bestVal == nxtVal)
            choices.push_back(nxt);
    }
    if (choices.size() == 0)
        return randomNeighbor(NI);
    return randomElement(choices);
}

int evnStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    // Want to maximize p_st = 1 - (1 - q_st)^k
    // k is out-degree of nxt
    // q_st is the probability of edge (nxt, dst) existing, given similarity(nxt, dst)

    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = -1.0;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = pst(G, nxt, dst);
        assert(nxtVal >= 0.0 && nxtVal <= 1.0);
        if (nxtVal > bestVal) {
            bestVal = nxtVal;
            choices.clear();
        }
        if (bestVal == nxtVal)
            choices.push_back(nxt);
    }
    if (choices.size() == 0)
        return randomNeighbor(NI);
    return randomElement(choices);
}

int similarityStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestSim = -1;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double sim = similarityCache[nxt][dst];
        if (bestSim < sim) {
            bestSim = sim;
            choices.clear();
        }
        if (bestSim == sim)
            choices.push_back(nxt);
    }
    if (choices.size() == 0)
        return randomNeighbor(NI);
    return randomElement(choices);
}

/* Returns the unvisited neighbor from a probability distribution
 * of nodes that are weighted by their degrees.
 * If all neighbors are visited, returns a random neighbor, or -1 if there are none
 */
int randomWeightedDegreeStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    vector<int> nodes, weights;
    int totalWeight = 0;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        int nxtDeg = G->GetNI(nxt).GetOutDeg();
        if (visited.find(nxt) == visited.end()) {
            nodes.push_back(nxt);
            weights.push_back(nxtDeg);
            totalWeight += nxtDeg;
        }
    }

    if (nodes.size() == 0)
        return randomNeighbor(NI);
    else
        best = selectWeightedNodes(nodes, weights, totalWeight);

    return best;
}


/*
 * Returns the unvisited neighbor with highest out-degree.
 * If all neighbors are visited, returns a random neighbor, or -1 if there are none.
 */
int degreeStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int bestDeg = -1;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        int nxtDeg = G->GetNI(nxt).GetOutDeg();
        if (bestDeg < nxtDeg) {
            bestDeg = nxtDeg;
            choices.clear();
        }
        if (bestDeg == nxtDeg)
            choices.push_back(nxt);
    }
    if (choices.size() == 0)
        return randomNeighbor(NI);
    return randomElement(choices);
}

/*
 * Returns a random unvisited neighbor.
 * If all neighbors are visited, returns a random neighbor, or -1 if there are none.
 */
int randomUnvisitedStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    vector<int> unvisited;
    TUNGraph::TNodeI NI = G->GetNI(cur);
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) == visited.end())
            unvisited.push_back(nxt);
    }
    if (unvisited.size() == 0)
        return randomNeighbor(NI);
    return unvisited[rand() % unvisited.size()];
}

/*
 * Returns a random neighbor, or -1 if there are none.
 */
int randomStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    return randomNeighbor(G->GetNI(cur));
}
