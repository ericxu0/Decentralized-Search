#include "Snap.h"
#include "spectra/SymEigsSolver.h"
#include "spectra/MatOp/SparseSymMatProd.h"
#include "utils.cpp"
#include <algorithm>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <vector>
using namespace std;
using namespace Eigen;
using namespace Spectra;

int spectralStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    double dist = 1E23;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        double d = getSpectralDist(nxt, dst);
        if (visited.find(nxt) == visited.end() && dist > d) {
            best = nxt;
            dist = d;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
}

int node2vecStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    double dist = 1E23;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        double d = getNode2VecDist(nxt, dst);
        if (visited.find(nxt) == visited.end() && dist > d) {
            best = nxt;
            dist = d;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
}

int similarityStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1, cnt = -1;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        int d = getSimilarity(nxt, dst);
        if (visited.find(nxt) == visited.end() && cnt < d) {
            best = nxt;
            cnt = d;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
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
    int best = -1, bestDeg = -1;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i), nxtDeg;
        if (visited.find(nxt) == visited.end() && bestDeg < (nxtDeg = G->GetNI(nxt).GetOutDeg())) {
            best = nxt;
            bestDeg = nxtDeg;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
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