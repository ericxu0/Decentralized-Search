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
#include <math.h>
#include <numeric>
#include <set>
#include <vector>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace Eigen;
using namespace Spectra;

map<int, double> clustCf;

double predictPathLength(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI curNI = G->GetNI(cur);
    int visitedNeighbors = 0;
    for (int i = 0; i < curNI.GetOutDeg(); i++) {
        int x = curNI.GetOutNId(i);
        visitedNeighbors += visited.find(x) != visited.end();
    }
    double fracVisited = 1.0 * visitedNeighbors / curNI.GetOutDeg();

    if (clustCf.find(cur) == clustCf.end())
        clustCf[cur] = TSnap::GetNodeClustCf(G, cur);

    vector<double> feat;
    //feat.push_back(G->GetNodes());                      // graph nodes
    //feat.push_back(G->GetEdges());                      // graph edges
    feat.push_back(getSimilarity(cur, dst));            // similarity
    feat.push_back(curNI.GetOutDeg());                  // degree
    feat.push_back(clustCf[cur]);                       // clustering coefficient
    feat.push_back(visited.find(cur) == visited.end()); // 1 if unvisited, 0 if visited
    feat.push_back(visitedNeighbors);                   // number of visited neighbors
    feat.push_back(fracVisited);                        // fraction of visited neighbors

    // pipe for tensorflow
    double weights[] = {-1.20790353e-01, -1.16367699e-03,  2.88583429e-01,  6.19656527e-02, 5.06394477e-02,  1.87175770e+00}; // TODO: for 107.edges
    //double weights[] = {-0.21352646,-0.01762945, 0.11491546,-0.31480496, 0.32982428, 0.3627876}; // TODO: for 0.edges

    double dot = 0.0;
    for (size_t i = 0; i < feat.size(); i++)
        dot += feat[i] * weights[i];
    return dot;
}



double tfPredictPathLength(PUNGraph& G, int cur, int dst, const set<int>& visited) {

    TUNGraph::TNodeI curNI = G->GetNI(cur);
    int visitedNeighbors = 0;
    for (int i = 0; i < curNI.GetOutDeg(); i++) {
        int x = curNI.GetOutNId(i);
        visitedNeighbors += visited.find(x) != visited.end();
    }
    double fracVisited = 1.0 * visitedNeighbors / curNI.GetOutDeg();

    if (clustCf.find(cur) == clustCf.end())
        clustCf[cur] = TSnap::GetNodeClustCf(G, cur);

    vector<double> feat;
    //feat.push_back(G->GetNodes());                      // graph nodes
    //feat.push_back(G->GetEdges());                      // graph edges
    feat.push_back(getSimilarity(cur, dst));            // similarity
    feat.push_back(curNI.GetOutDeg());                  // degree
    feat.push_back(clustCf[cur]);                       // clustering coefficient
    feat.push_back(visited.find(cur) == visited.end()); // 1 if unvisited, 0 if visited
    feat.push_back(visitedNeighbors);                   // number of visited neighbors
    feat.push_back(fracVisited);                        // fraction of visited neighbors



    //feat.insert(feat.end(), node2vec_embeddings[cur].begin(), node2vec_embeddings[cur].end());

    int lengthVector = feat.size();

    printf("%d\n", lengthVector);

    for(int i = 0; i < lengthVector; i++) {
        printf("%f\n", feat[i]);
    }

    fflush(stdout);

    double prediction;
    scanf("%lf", &prediction);

    // fprintf(stderr, "Predicted Path: %lf\n", prediction);

    return prediction;
    
}

int NeuralNetStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {

    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = 1E20;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = tfPredictPathLength(G, nxt, dst, visited);
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

int LinRegStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
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


// TODO: this is incredibly hacky
// Basically we need to compute, for all similarity values V, the value:
//   [# node-pairs (s,t) where (s,t) is an edge and similarity(s,t)=V] / [# pairs (s,t) where similarity(s,t)=V]
// which also depends on the graph(s) under consideration.
map<int, double> init() {
    map<int, double> map;

    // On facebook 0.edges
    /*
    map[0 ] = 0.0298034242232;
    map[1 ] = 0.0361002349256;
    map[2 ] = 0.0396181384248;
    map[3 ] = 0.0194543007464;
    map[4 ] = 0.0314450018353;
    map[5 ] = 0.0575401069519;
    map[6 ] = 0.0813815005955;
    map[7 ] = 0.131618759455 ;
    map[8 ] = 0.190409026798 ;
    map[9 ] = 0.268571428571 ;
    map[10] = 0.331550802139 ;
    map[11] = 0.434343434343 ;
    map[12] = 0.5            ;
    map[13] = 0.6            ;
    map[14] = 0.5            ;
    map[15] = 1.0            ;
    */

    // On facebook 107.edges
    map[0 ] = 0.0111970622761;
    map[1 ] = 0.0283930565425;
    map[2 ] = 0.0294564391885;
    map[3 ] = 0.0268360588182;
    map[4 ] = 0.0519978225367;
    map[5 ] = 0.0932846354718;
    map[6 ] = 0.140991292699 ;
    map[7 ] = 0.193237158872 ;
    map[8 ] = 0.254996205414 ;
    map[9 ] = 0.323699421965 ;
    map[10] = 0.342592592593 ;
    map[11] = 0.541353383459 ;
    map[12] = 0.4            ;
    map[13] = 0.538461538462 ;
    map[14] = 0.0            ;
    map[15] = 1.0            ;
    map[16] = 0              ;
    map[17] = 0              ;
    map[18] = 1.0            ;
    return map;
}
map<int, double> mapSimilarityToEdgeProbability = init();

double q(int sim) {
    return mapSimilarityToEdgeProbability[sim];
}

int EVNStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    // We want to maximize p_st = 1 - (1 - q_st)^k
    // Equivalent to minimizing (1 - q_st)^k
    // Equivalent to minimizing log(1 - q_st) * k
    // k is out-degree of nxt
    // q_st is the probability of edge (nxt, dst) existing, given similarity(nxt, dst)

    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = 2.0;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = log(1.0 - q(getSimilarity(nxt, dst))) * G->GetNI(nxt).GetOutDeg(); // log(1 - q_st)^k
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
    int bestSim = -1;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        int sim = getSimilarity(nxt, dst);
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
