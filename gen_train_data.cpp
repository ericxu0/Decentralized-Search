#include "Snap.h"
#include "utils.cpp"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

using namespace std;

const bool DEBUG = false;
const bool WRITE_TO_FILE = true;
const string DATA_PREFIX = "data/training_data/training_data_";

const string GRAPH_EXTENSION = ".edges";

const int NUM_SAMPLES = 10000;
const int NUM_WALKS = 1000;
const int TRAIN_SEED = 24;
const double PROB_RANDOM = 1.0/3;

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

void getTrainingData(const string& filename, const string& graphName, bool isCitation) {
    cout << "\nGenerating training data on " << filename << " (graphName = " << graphName << ")" << endl;
    PUNGraph G = loadGraph(filename);

    map<int, int> compIdx;
    computeCompressionIndices(G, compIdx);

    vector<vector<int> > minDist;
    computeShortestPaths(G, compIdx, minDist);
    computeClusterCf(G);
    generateNode2vecEmbeddings(filename);
    generateSimilarityFeatures(filename, isCitation);

    vector<pair<int, int> > samples;
    srand(TRAIN_SEED);
    getSamples(G, samples, NUM_SAMPLES);

    for (int ii = 0; ii < 2; ii++) {
        bool node2vec = (ii == 1);
        if (node2vec)
            normalizeSimilarity(G, SIMILARITY_NODE2VEC);
        else
            normalizeSimilarity(G, isCitation ? SIMILARITY_TFIDF : SIMILARITY_FEATURES);
        computeDiscretizedQ(G);

        ofstream dataFile;
        string path = DATA_PREFIX + graphName + (node2vec ? "_n2v" : "_sim") + ".txt";
        dataFile.open(path);

        srand(TRAIN_SEED); // Make performWalk() identical regardless of ii
        for (size_t i = 0; i < samples.size(); i++) {
            auto& s = samples[i];

            vector<int> path;
            performWalk(G, compIdx, minDist, s.first, s.second, path);
            set<int> visited;
            int pidx = rand() % path.size();
            for (int i = 0; i <= pidx; i++)
                visited.insert(path[i]);

            vector<double> feat;
            getFeatureVector(G, randomNeighbor(G->GetNI(path[pidx])), s.second, visited, feat);

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
        dataFile.close();
        cout << "Wrote training data to file: " << path << endl;
    }

    cout << "Finished generating training data." << endl;
}

int main() {
    string facebookRoot = "data/real/facebook/";
    vector<string> allEdgeFiles = getAllFiles(facebookRoot, GRAPH_EXTENSION);
    for(auto&& fileName : allEdgeFiles) {
        string fullFileName = facebookRoot + fileName;
        getTrainingData(fullFileName, "facebook-" + getBase(fileName), false);
    }
    getTrainingData("data/real/cit-HepTh/cit-HepTh-subset.edges", "cit-HepTh-subset", true);

    return 0;
}
