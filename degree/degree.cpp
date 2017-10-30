#include "Snap.h"
#include <iostream>
#include <algorithm>
#include <vector>
#include <random>
#include <set>
using namespace std;

const int NUM_TRIALS = 1000;
const int SEED = 42;
const int CAP = 1000000000;

int getNextNode(PUNGraph& G, int cur, set<int>& visited) {
    TUNGraph::TNodeI NI = G->GetNI(cur);
    int best = -1;
    for (int i = 0; i < NI.GetOutDeg(); i++) {
        int nxt = NI.GetOutNId(i);
        if (visited.find(nxt) == visited.end() && (best == -1 || G->GetNI(best).GetOutDeg() < G->GetNI(nxt).GetOutDeg()))
            best = nxt;
    }
    if (best == -1)
        return NI.GetOutNId(rand() % NI.GetOutDeg());
    return best;
}

int search(PUNGraph& G, int src, int dst) {
    int dist = 0, cur = src;
    set<int> visited;
    while (cur != dst) {
        visited.insert(cur);
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

void simulate(PUNGraph& G) {
    vector<int> node;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        node.push_back(NI.GetId());
    int N = node.size();

    int numSuccess = 0, totalPathLength = 0;
    srand(SEED);
    for (int i = 0; i < NUM_TRIALS;) {
        int src = rand() % N;
        int dst = rand() % N;
        if (src != dst) {
            int dist = search(G, node[src], node[dst]);
            if (dist != -1) {
                numSuccess++;
                totalPathLength += dist;
            }
            i++;
        }
    }
    
    if (numSuccess == 0)
        cout << "No successes\n";
    else
        cout << "Success rate: " << 1.0*numSuccess/NUM_TRIALS << "\nAverage path length: " << 1.0*totalPathLength/numSuccess << "\n";
}

int main() {
    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>("facebook_combined.txt", 0, 1);
    simulate(G);
    return 0;
}
