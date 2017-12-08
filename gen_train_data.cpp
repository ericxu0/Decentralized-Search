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

const bool DEBUG = false;
const bool WRITE_TO_FILE = true;

const string GRAPH_EXTENSION = ".edges";

const int NUM_SAMPLES = 1000000;
const int SEED = 224;
const int INFTY = 1<<28;
const double probRandomEdge = 1.0/3;

void getFeatureVector(PUNGraph& G, int cur, int dst, set<int>& visited, vector<double>& feat, map<int, int>& visitedCnt) {
    TUNGraph::TNodeI curNI = G->GetNI(cur);
    int visitedNeighbors = 0;
    for (int i = 0; i < curNI.GetOutDeg(); i++) {
        int x = curNI.GetOutNId(i);
        visitedNeighbors += visited.find(x) != visited.end();
    }
    visitedCnt[visitedNeighbors]++;

    feat.push_back(G->GetNodes());                      // graph nodes
    feat.push_back(G->GetEdges());                      // graph edges
    feat.push_back(getSimilarity(cur, dst));            // similarity
    feat.push_back(curNI.GetOutDeg());                  // degree
    feat.push_back(visited.find(cur) == visited.end()); // 0 if unvisited, 1 if visited
    feat.push_back(visitedNeighbors);                   // number of visited neighbors
    feat.push_back(0);                                  // TODO: abs(v2[cur] - v2[dst]), where v2 is the 2nd eigenvector
    feat.push_back(0);                                  // TODO: ...
    feat.push_back(0);                                  // TODO: abs(v6[cur] - v6[dst])
    feat.push_back(0);                                  // TODO: something with node2vec
    feat.push_back(1);                                  // constant term for linear regression
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
        int nxtDist;
        vector<int> choices;
        if (1.0*rand()/RAND_MAX <= probRandomEdge)
            nxt = randomNeighbor(curNI);
        else {
            nxt = curNI.GetOutNId(0);
            choices.push_back(nxt);
            nxtDist = minDist[compIdx[dst]][compIdx[nxt]];
            for (int i = 1; i < curNI.GetOutDeg(); i++) {
                int x = curNI.GetOutNId(i);
                int xDist = minDist[compIdx[dst]][compIdx[x]];
                if (xDist < nxtDist) {
                    nxt = x;
                    nxtDist = xDist;
                    choices.clear();
                }
                if (xDist == nxtDist) {
                    choices.push_back(x);
                }
            }
            nxt = choices[rand() % choices.size()];
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

void getTrainingData(const string& filename, ofstream& dataFile) {
    cout << "\nGenerating training Data on " << filename << endl;
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
    for (size_t i = 0; i < samples.size(); i++) {
        auto& s = samples[i];
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

        if (DEBUG) {
            cout << "Training pair " << i+1 << ":" << endl;
            cout << "  Input (feature vector):   ";
            for (int f : feat)
                cout << " " << f;
            cout << endl;
            cout << "  Output (true path length): " << minDist[a][b] << endl;
        } else if ((i+1) % (NUM_SAMPLES / 10) == 0) {
            cout << "Finished generating training pair " << (i+1) << endl;
        }

        if (WRITE_TO_FILE) {
            for (int f : feat)
                dataFile << f << " ";
            dataFile << ", " << minDist[a][b] << endl;
        }
    }

    if (DEBUG) {
        cout << "True Path Lengths:\n";
        for (auto& e : truePathLen)
            cout << e.first << ": " << e.second << "\n";
        cout << "Generated Path Lengths:\n";
        for (auto& e : genPathLen)
            cout << e.first << ": " << e.second << "\n";
        cout << "Number of Visited Neighbors:\n";
        for (auto& e : visitedCnt)
            cout << e.first << ": " << e.second << "\n";
    } else {
        cout << "Finished generating training data." << endl;
    }
}

int main() {
    ofstream dataFile;
    if (WRITE_TO_FILE)
        dataFile.open("training_data.txt");

    // TODO: do this for all graphs, should programmatically find all files like "data/real/*/*.edges"

    //getTrainingData("data/real/gplus/", dataFile);
    //getTrainingData("data/real/twitter/", dataFile);

    string facebookRoot = "data/real/facebook/";
    vector<string> allEdgeFiles = getAllFiles(facebookRoot, GRAPH_EXTENSION);
    for(auto&& fileName : allEdgeFiles) {
        string fullFileName = facebookRoot + fileName;
        getTrainingData(fullFileName, dataFile);
    }

    if (WRITE_TO_FILE)
        dataFile.close();
    return 0;
}
