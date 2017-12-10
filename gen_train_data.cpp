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
const string DATA_PREFIX = "data/training_data/training_data_10k_";

const string GRAPH_EXTENSION = ".edges";

const int NUM_SAMPLES = 10000;
const int NUM_WALKS = 1000;
const int SEED = 224;
const double PROB_RANDOM = 1.0/3;

void getFeatureVector(PUNGraph& G, int cur, int dst, set<int>& visited, map<int, double>& clusterCf, vector<double>& feat, double isCitation, double avg_path_len) {
    TUNGraph::TNodeI curNI = G->GetNI(cur);
    int visitedNeighbors = 0;
    for (int i = 0; i < curNI.GetOutDeg(); i++) {
        int x = curNI.GetOutNId(i);
        visitedNeighbors += visited.find(x) != visited.end();
    }
    double fracVisited = 1.0 * visitedNeighbors / curNI.GetOutDeg();

    //feat.push_back(G->GetNodes());                      // graph nodes
    //feat.push_back(G->GetEdges());                      // graph edges
    feat.push_back(avg_path_len);
    feat.push_back(getSimilarity(cur, dst, isCitation));  // similarity
    feat.push_back(curNI.GetOutDeg());                  // degree
    feat.push_back(clusterCf[cur]);                     // clustering coefficient
    feat.push_back(visited.find(cur) == visited.end()); // 1 if unvisited, 0 if visited
    //feat.push_back(visitedNeighbors);                   // number of visited neighbors
    feat.push_back(fracVisited);                        // fraction of visited neighbors
    feat.push_back(0); //TODO: EVN Probability
    //feat.push_back(getNode2VecL1Dist(cur, dst)); // L1 distance of node2vec
    //feat.push_back(getNode2VecL2Dist(cur, dst)); // L2 distance of node2vec
    //feat.push_back(getNode2VecLInfDist(cur, dst)); // LInfinity distance of node2vec
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
        if (1.0*rand()/RAND_MAX <= PROB_RANDOM)
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
            nxt = randomElement(choices);
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

void getTrainingData(const string& filename, ofstream& dataFile, bool isCitation) {
    cout << "\nGenerating training data on " << filename << endl;
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
    double avg_path_len = computeShortestPath(G, compIdx, minDist);
    
    //spectral_embeddings.clear();
    //generateSpectralEmbeddings(G, compIdx);
    node2vec_embeddings.clear();
    generateNode2vecEmbeddings(filename);
    generateSimilarityFeatures(filename, isCitation);
    clearSimilarityCache();
    normalizeSimilarity(G, isCitation);

    map<int, double> clusterCf;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        clusterCf[NI.GetId()] = TSnap::GetNodeClustCf(G, NI.GetId());

    vector<pair<int, int> > samples;
    getSamples(G, samples);
    for (size_t i = 0; i < samples.size(); i++) {
        auto& s = samples[i];

        vector<int> path;
        performWalk(G, compIdx, minDist, s.first, s.second, path);
        set<int> visited;
        int pidx = rand() % path.size();
        for (int i = 0; i <= pidx; i++)
            visited.insert(path[i]);
        
        vector<double> feat;
        getFeatureVector(G, randomNeighbor(G->GetNI(path[pidx])), s.second, visited, clusterCf, feat, isCitation, avg_path_len);

        int total = 0;
        for (int j = 0; j < NUM_WALKS; j++) {
            vector<int> walk;
            performWalk(G, compIdx, minDist, path[pidx], s.second, walk);
            total += walk.size() - 1;
        }
        double avgRandomWalkLength = 1.0 * total / NUM_WALKS;
        double truePathLength = minDist[compIdx[path[pidx]]][compIdx[s.second]];

        if (DEBUG) {
            cout << "Training pair " << i+1 << ":" << endl;
            cout << "  Input (feature vector):";
            for (double f : feat)
                cout << " " << f;
            cout << endl;
            cout << "  Output (true path length): " << truePathLength << endl;
            cout << "  Output (avg random walk length): " << avgRandomWalkLength << endl;
        } else if ((i+1) % (NUM_SAMPLES / 10) == 0) {
            cout << "Finished generating " << (i+1) << " training pairs" << endl;
        }

        if (WRITE_TO_FILE) {
            string sep = "";
            for (double f : feat) {
                dataFile << sep << f;
                sep = " ";
            }
            dataFile << ", " << truePathLength << ", " << avgRandomWalkLength << endl;
        }
    }

    cout << "Finished generating training data." << endl;
}

int main() {
    //getTrainingData("data/real/gplus/", dataFile);
    //getTrainingData("data/real/twitter/", dataFile);

    string facebookRoot = "data/real/facebook/";
    vector<string> allEdgeFiles = getAllFiles(facebookRoot, GRAPH_EXTENSION);
    for(auto&& fileName : allEdgeFiles) {
        string fullFileName = facebookRoot + fileName;
        if (WRITE_TO_FILE) {
            ofstream dataFile;
            string path = DATA_PREFIX + "facebook_" + getBase(fileName) + ".txt";
            dataFile.open(path);
            getTrainingData(fullFileName, dataFile, false);
            dataFile.close();
            cout << "Wrote training data to file: " << path << endl;
        }
    }
    if (WRITE_TO_FILE) {
        ofstream dataFile;
        string path = DATA_PREFIX + "cit-HepTh-subset.txt";
        dataFile.open(path);
        getTrainingData("data/real/cit-HepTh/cit-HepTh-subset.edges", dataFile, true);
        dataFile.close();
        cout << "Wrote training data to file: " << path << endl;
    }

    return 0;
}
