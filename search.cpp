#include "Snap.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>
using namespace std;

const int NUM_TRIALS = 1000;
const int SEED = 42;
const int CAP = 100 * 1000;
const int BUCKETS[] = {10, 100, 1000, 10000};

// Returns a random neighbor, or -1 if there are none.
int randomNeighbor(TUNGraph::TNodeI NI) {
    if (NI.GetOutDeg() == 0)
        return -1;
    return NI.GetOutNId(rand() % NI.GetOutDeg());
}

/*
 * Returns the unvisited neighbor with highest out-degree.
 * If all neighbors are visited, returns a random neighbor, or -1 if there are none.
 */
int degreeStrategy(PUNGraph& G, int cur, const set<int>& visited) {
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
int randomUnvisitedStrategy(PUNGraph& G, int cur, const set<int>& visited) {
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
int randomStrategy(PUNGraph& G, int cur, const set<int>& visited) {
    return randomNeighbor(G->GetNI(cur));
}

// Returns the number of steps, or -1 if failure.
int search(PUNGraph& G, int src, int dst, int (*getNextNode)(PUNGraph&, int, const set<int>&)) {
    int dist = 0, cur = src;
    set<int> visited;
    while (cur != dst) {
        visited.insert(cur);
        
        if (G->IsEdge(cur, dst))
            return dist + 1;
        
        int nxt = getNextNode(G, cur, visited);
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
        cout << "Length <= " << setw(5) << size << ": " << fixed << setprecision(1) << setw(4) << 100.0 * count / NUM_TRIALS << "\%\n";
    }

    cout << endl;
}

void simulate(PUNGraph& G, vector<pair<int, int> >& samples, int (*getNextNode)(PUNGraph&, int, const set<int>&)) {
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

void experiment(const string& filename) {
    cout << "Running experiment on " << filename << endl;
    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>(filename.c_str(), 0, 1);
    cout << "# Nodes: " << G->GetNodes() << endl;
    cout << "# Edges: " << G->GetEdges() << endl;
    G = TSnap::GetMxWcc(G);
    cout << "# Nodes (Max WCC): " << G->GetNodes() << endl;
    cout << "# Edges (Max WCC): " << G->GetEdges() << endl;
    cout << endl;

    vector<pair<int, int> > samples;
    getSamples(G, samples);

    cout << "Simulating degree strategy\n";
    simulate(G, samples, degreeStrategy);

    cout << "Simulating random unvisited strategy\n";
    simulate(G, samples, randomUnvisitedStrategy);

    cout << "Simulating random strategy\n";
    simulate(G, samples, randomStrategy);

    cout << "Optimal\n";
    optimal(G, samples);

    cout << endl;
}

int main() {
    experiment("data/real/facebook_combined.txt");
    experiment("data/real/ca-HepTh.txt");
    experiment("data/real/cit-HepTh.txt");

    experiment("data/synthetic/gnm0.txt");
    experiment("data/synthetic/smallworld0.txt");
    experiment("data/synthetic/prefattach0.txt");

    return 0;
}
