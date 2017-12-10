#include "Snap.h"
#include "strategies.cpp"
#include <algorithm>
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

const string GRAPH_EXTENSION = ".edges";
const int NUM_TRIALS = 10000;
const int TEST_SEED = 42;
const int CAP = 100;
const int BUCKETS[] = {3, 10, 30, 100};
const bool PRINT_BUCKETS = false;

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
        cout << "  No successes\n";
    else
        cout << "  Success rate: " << 100.0 * numSuccess / NUM_TRIALS << "\%\n  Average path length: " << 1.0 * totalPathLength / numSuccess << "\n";

    if (PRINT_BUCKETS) {
        for (int size : BUCKETS) {
            int count = 0;
            for (int val : results)
                if (val <= size)
                    count++;
            ios init(NULL);
            init.copyfmt(cout);
            cout << "  Length <= " << setw(5) << size << ": " << fixed << setprecision(1) << setw(4) << 100.0 * count / NUM_TRIALS << "\%\n";
            cout.copyfmt(init);
        }
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
    printf("Finished Simulation\n");
    fflush(stdout);
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

void experiment(const string& filename, bool isCitation) {
    cout << "Running experiment on " << filename << endl;
    PUNGraph G = loadGraph(filename);

    map<int, int> compIdx;
    computeCompressionIndices(G, compIdx);

    vector<vector<int> > minDist;
    computeShortestPaths(G, compIdx, minDist);
    computeClusterCf(G);
    generateNode2vecEmbeddings(filename);
    generateSimilarityFeatures(filename, isCitation);

    normalizeSimilarity(G, isCitation ? SIMILARITY_TFIDF : SIMILARITY_FEATURES);
    computeDiscretizedQ(G);
    getRegressionWeights(filename, isCitation, false);

    vector<pair<int, int> > samples;
    srand(TEST_SEED);
    getSamples(G, samples, NUM_TRIALS);
    
    // cout << "Random unvisited\n";
    // simulate(G, samples, randomUnvisitedStrategy);

    // cout << "Degree\n";
    // simulate(G, samples, degreeStrategy);

    // cout << "Similarity\n";
    // simulate(G, samples, similarityStrategy);

    // cout << "EVN (with similarity)\n";
    // simulate(G, samples, evnStrategy);

    // cout << "Ridge regression (with similarity)\n";
    // simulate(G, samples, ridgeStrategy);

    // cout << "Overall ridge regression (with similarity)\n";
    // simulate(G, samples, overallRidgeStrategy);

    // Replacing similarity with node2vec
    normalizeSimilarity(G, SIMILARITY_NODE2VEC);
    computeDiscretizedQ(G);
    getRegressionWeights(filename, isCitation, true);

    // cout << "node2vec L_2\n";
    // simulate(G, samples, similarityStrategy);

    // cout << "EVN (with node2vec)\n";
    // simulate(G, samples, evnStrategy);

    // cout << "Ridge regression (with node2vec)\n";
    // simulate(G, samples, ridgeStrategy);

    // cout << "Overall ridge regression (with node2vec)\n";
    // simulate(G, samples, overallRidgeStrategy);

    printf("Starting Strategy\n");
    fflush(stdout);

    simulate(G, samples, neuralNetStrategy);

    cout << "Optimal\n";
    optimal(G, samples);

    printf("Finished Experiment\n");
    fflush(stdout);

    cout << endl;
}

int main() {
    experiment("data/real/cit-HepTh/cit-HepTh-subset.edges", true);

    string facebookRoot = "data/real/facebook/";
    vector<string> allEdgeFiles = getAllFiles(facebookRoot, GRAPH_EXTENSION);
    for(auto&& fileName : allEdgeFiles) {
        string fullFileName = facebookRoot + fileName;
        if (fileName == "0.edges")
            experiment(fullFileName, false);

    }

    return 0;
}