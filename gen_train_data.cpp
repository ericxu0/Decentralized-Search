#include "Snap.h"
#include "spectra/SymEigsSolver.h"
#include "spectra/MatOp/SparseSymMatProd.h"
#include "utils.cpp"
#include <algorithm>
#include <cassert>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>
using namespace std;
using namespace Eigen;
using namespace Spectra;

const int NUM_SAMPLES = 100000;
const int SEED = 224;
const int INFTY = 1<<28;
const double probRandomEdge = 0.3;

void getFeatureVector(PUNGraph &G, int cur, int dst, set<int>& visited, vector<double>& feat, map<int, int>& visitedCnt) {
    TUNGraph::TNodeI curNI = G->GetNI(cur);
    int visitedNeighbors = 0;
    for (int i = 1; i < curNI.GetOutDeg(); i++) {
        int x = curNI.GetOutNId(i);
        visitedNeighbors += visited.find(x) != visited.end();
    }
    visitedCnt[visitedNeighbors]++;

    feat.push_back(getSimilarity(cur, dst)); //similarity
    feat.push_back(curNI.GetOutDeg()); //degree
    feat.push_back(visited.find(cur) == visited.end()); //unvisited
    feat.push_back(visitedNeighbors); //visited neighbors count
}

void performWalk(PUNGraph& G, map<int, int>& compIdx, vector<vector<int> >& minDist, int src, int dst, vector<int>& path) {
    int cur = src;
    path.push_back(cur);
    while (cur != dst) {
        if (G->IsEdge(cur, dst)) {
            path.push_back(dst);
            cur = dst;
            continue;
        }
        
        TUNGraph::TNodeI curNI = G->GetNI(cur);
        int nxt;
        if (1.0*rand()/RAND_MAX <= probRandomEdge)
            nxt = randomNeighbor(curNI);
        else {
            nxt = curNI.GetOutNId(0);
            for (int i = 1; i < curNI.GetOutDeg(); i++) {
                int x = curNI.GetOutNId(i);
                if (minDist[compIdx[x]][compIdx[dst]] < minDist[compIdx[nxt]][compIdx[dst]])
                    nxt = x;
            }
        }
    
        path.push_back(nxt);
        cur = nxt;
    }
}

void getSamples(PUNGraph& G, vector<pair<int, int> >& samples) {
    vector<int> nodes;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        nodes.push_back(NI.GetId());
    int N = nodes.size();

    srand(SEED);
    for (int i = 0; i < NUM_SAMPLES; ) {
        int src = nodes[rand() % N];
        int dst = nodes[rand() % N];
        if (src != dst) {
            samples.push_back(make_pair(src, dst));
            i++;
        }
    }
}

void computeShortestPath(PUNGraph& G, map<int, int>& compIdx, vector<vector<int> >& minDist) {
    int N = G->GetNodes();
    for (int i = 0; i < N; i++)
        minDist.push_back(vector<int>(N, INFTY));
    for (int i = 0; i < N; i++)
        minDist[i][i] = 0;
    for (TUNGraph::TEdgeI EI = G->BegEI(); EI < G->EndEI(); EI++) {
        int a = compIdx[EI.GetSrcNId()];
        int b = compIdx[EI.GetDstNId()];
        minDist[a][b] = minDist[b][a] = 1;//undirected
    }
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                minDist[i][j] = min(minDist[i][j], minDist[i][k] + minDist[k][j]);
}

void getTrainingData(const string& filename) {
    cout << "Generating Training Data on " << filename << endl;
    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>(filename.c_str(), 0, 1);
    cout << "# Nodes: " << G->GetNodes() << endl;
    cout << "# Edges: " << G->GetEdges() << endl;
    G = TSnap::GetMxWcc(G);
    cout << "# Nodes (Max WCC): " << G->GetNodes() << endl;
    cout << "# Edges (Max WCC): " << G->GetEdges() << endl;
    cout << endl;

    map<int, int> compIdx;
    int index = 0;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        compIdx[NI.GetId()] = index++;

    vector<vector<int> > minDist;
    computeShortestPath(G, compIdx, minDist);
    
    //spectral_embeddings.clear();
    //generateSpectralEmbeddings(G, compIdx);
    //node2vec_embeddings.clear();
    //generateNode2vecEmbeddings(filename);
    similarity_features.clear();
    generateSimilarityFeatures(filename);

    vector<pair<int, int> > samples;
    getSamples(G, samples);
    map<int, int> truePathLen, genPathLen, visitedCnt;
    for (auto& s : samples) {
        int a = compIdx[s.first];
        int b = compIdx[s.second];
        truePathLen[minDist[a][b]]++;

        vector<int> path;
        performWalk(G, compIdx, minDist, s.first, s.second, path);
        genPathLen[(int)path.size() - 1]++;
        set<int> visited;
        int pidx = rand() % path.size();
        for (int i = 0; i <= pidx; i++)
            visited.insert(path[i]);
        
        vector<double> feat;
        getFeatureVector(G, randomNeighbor(G->GetNI(path[pidx])), s.second, visited, feat, visitedCnt);
    }

    cout << "True Path Lengths:\n";
    for (auto& e : truePathLen)
        cout << e.first << ": " << e.second << "\n";
    cout << "Generated Path Lengths:\n";
    for (auto& e : genPathLen)
        cout << e.first << ": " << e.second << "\n";
    cout << "Number of Visited Neighbors:\n";
    for (auto& e : visitedCnt)
        cout << e.first << ": " << e.second << "\n";
}

int main() {
    getTrainingData("data/real/facebook/0.edges");
    //getTrainingData("data/real/gplus/");
    //getTrainingData("data/real/twitter/");

    return 0;
}
