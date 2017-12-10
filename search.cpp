#include "Snap.h"
#include "spectra/SymEigsSolver.h"
#include "spectra/MatOp/SparseSymMatProd.h"
#include "strategies.cpp"
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
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace Eigen;
using namespace Spectra;

const string GRAPH_EXTENSION = ".edges";
const int NUM_TRIALS = 10000;
const int SEED = 42;
const int CAP = 100;
const int BUCKETS[] = {3, 10, 30, 100};

// Returns the number of steps, or -1 if failure.
int search(PUNGraph& G, int src, int dst, int (*getNextNode)(PUNGraph&, int, int, const set<int>&)) {
    int dist = 0, cur = src;
    set<int> visited;
    while (cur != dst) {
        visited.insert(cur);
        
        if (G->IsEdge(cur, dst))
            return dist + 1;
        
        int nxt = getNextNode(G, cur, dst, visited);
        if (nxt == -1)
            return -1;
        cur = nxt;
        dist++;
        if (dist == CAP)
            return -1;
    }
    return dist;
}

void displayResults(vector<int>& results) {
    int numSuccess = results.size();
    int totalPathLength = accumulate(results.begin(), results.end(), 0);
    
    if (numSuccess == 0)
        cout << "No successes\n";
    else
        cout << "Success rate: " << 100.0 * numSuccess / NUM_TRIALS << "\%\nAverage path length: " << 1.0 * totalPathLength / numSuccess << "\n";

    for (int size : BUCKETS) {
        int count = 0;
        for (int val : results)
            if (val <= size)
                count++;
        ios init(NULL);
        init.copyfmt(cout);
        cout << "Length <= " << setw(5) << size << ": " << fixed << setprecision(1) << setw(4) << 100.0 * count / NUM_TRIALS << "\%\n";
        cout.copyfmt(init);
    }

    cout << endl;
}

void simulate(PUNGraph& G, vector<pair<int, int> >& samples, int (*getNextNode)(PUNGraph&, int, int, const set<int>&)) {
    vector<int> results;
    for (size_t i = 0; i < samples.size(); i++) {
        int dist = search(G, samples[i].first, samples[i].second, getNextNode);
        if (dist != -1)
            results.push_back(dist);
    }
    displayResults(results);
}

void optimal(PUNGraph& G, vector<pair<int, int> >& samples) {
    vector<int> results;
    for (size_t i = 0; i < samples.size(); i++) {
        int dist = TSnap::GetShortPath(G, samples[i].first, samples[i].second);
        if (dist != -1)
            results.push_back(dist);
    }
    displayResults(results);
}

void getSamples(PUNGraph& G, vector<pair<int, int> >& samples) {
    vector<int> nodes;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        nodes.push_back(NI.GetId());
    int N = nodes.size();

    srand(SEED);
    for (int i = 0; i < NUM_TRIALS; ) {
        int src = nodes[rand() % N];
        int dst = nodes[rand() % N];
        if (src != dst) {
            samples.push_back(make_pair(src, dst));
            i++;
        }
    }
}

void experiment(const string& filename, bool isCitation) {
    cout << "Running experiment on " << filename << endl;
    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>(filename.c_str(), 0, 1);
    cout << "# Nodes: " << G->GetNodes() << endl;
    cout << "# Edges: " << G->GetEdges() << endl;
    G = TSnap::GetMxWcc(G);
    TSnap::DelSelfEdges(G);
    cout << "# Nodes (Max WCC): " << G->GetNodes() << endl;
    cout << "# Edges (Max WCC): " << G->GetEdges() << endl;
    cout << endl;

    map<int, int> compIdx;
    int index = 0;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        compIdx[NI.GetId()] = index++;

    vector<vector<int> > minDist;
    AVG_PATH_LEN = computeShortestPath(G, compIdx, minDist);

    //spectral_embeddings.clear();
    //generateSpectralEmbeddings(G, compIdx);
    node2vec_embeddings.clear();
    generateNode2vecEmbeddings(filename);
    generateSimilarityFeatures(filename, isCitation);
    getRegressionWeights(filename, isCitation);

    IS_CITATION = isCitation;
    clearSimilarityCache();
    normalizeSimilarity(G, isCitation);
    computeDiscretizedQ(G);

    vector<pair<int, int> > samples;
    getSamples(G, samples);
    
    cout << "Random unvisited\n";
    simulate(G, samples, randomUnvisitedStrategy);

    cout << "Degree\n";
    simulate(G, samples, degreeStrategy);

    cout << "Similarity\n";
    simulate(G, samples, similarityStrategy);

    cout << "EVN (with similarity)\n";
    simulate(G, samples, evnStrategy);

    cout << "Ridge regression (with similarity)\n";
    simulate(G, samples, ridgeStrategy); // TODO: currently uses node2vec features?

    cout << "node2vec L_2\n";
    simulate(G, samples, node2vecL2Strategy);

    cout << "EVN (with node2vec)\n";
    // TODO

    cout << "Ridge regression (with node2vec)\n";
    // TODO

    cout << "Optimal\n";
    optimal(G, samples);

    cout << endl;
}

int main() {
    //experiment("data/real/facebook_combined.txt");
    //experiment("data/real/ca-HepTh.txt");
    //experiment("data/real/cit-HepTh.txt");

    //experiment("data/synthetic/gnm0.txt");
    //experiment("data/synthetic/smallworld0.txt");
    //experiment("data/synthetic/powerlaw0.txt");
    //experiment("data/synthetic/prefattach0.txt");
    
    //experiment("data/synthetic/gnm_small0.txt");
    //experiment("data/synthetic/smallworld_small0.txt");
    //experiment("data/synthetic/powerlaw_small0.txt");
    //experiment("data/synthetic/prefattach_small0.txt");
    
    //experiment("data/real/facebook/0.edges");

    string facebookRoot = "data/real/facebook/";
    vector<string> allEdgeFiles = getAllFiles(facebookRoot, GRAPH_EXTENSION);
    for(auto&& fileName : allEdgeFiles) {
        string fullFileName = facebookRoot + fileName;
        if (fileName == "0.edges")
            experiment(fullFileName, false);
    }

    experiment("data/real/cit-HepTh/cit-HepTh-subset.edges", true);

    return 0;
}
