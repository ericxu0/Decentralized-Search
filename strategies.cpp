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

bool IS_CITATION = false;

double predictPathLength(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    vector<double> feat;
    getFeatureVector(G, cur, dst, visited, feat);

    assert(feat.size() == ridgeWeights.size());
    double dot = 0.0;
    for (size_t i = 0; i < feat.size(); i++)
        dot += feat[i] * ridgeWeights[i];
    return dot;
}

/*
int lassoStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = 1E20;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = predictPathLength(G, nxt, dst, visited, "Lasso");
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

int olsStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = 1E20;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = predictPathLength(G, nxt, dst, visited, "OLS");
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

int elasticNetStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    double bestVal = 1E20;
    vector<int> choices;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) != visited.end())
            continue;
        double nxtVal = predictPathLength(G, nxt, dst, visited, "ElasticNet");
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
*/

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

/*
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
    feat.push_back(getSimilarity(cur, dst, IS_CITATION)); // similarity
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
*/

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

/*
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

int node2vecL1Strategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    double dist = 1E23;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        double d = getNode2VecL1Dist(nxt, dst);
        if (visited.find(nxt) == visited.end() && dist > d) {
            best = nxt;
            dist = d;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
}

int node2vecL2Strategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    double dist = 1E23;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        double d = getNode2VecL2Dist(nxt, dst);
        if (visited.find(nxt) == visited.end() && dist > d) {
            best = nxt;
            dist = d;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
}

int node2vecLInfStrategy(PUNGraph& G, int cur, int dst, const set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    double dist = 1E23;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        double d = getNode2VecLInfDist(nxt, dst);
        if (visited.find(nxt) == visited.end() && dist > d) {
            best = nxt;
            dist = d;
        }
    }
    if (best == -1)
        return randomNeighbor(NI);
    return best;
}
*/

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
