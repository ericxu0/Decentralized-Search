#include "Snap.h"
#include "spectra/SymEigsSolver.h"
#include "spectra/MatOp/SparseSymMatProd.h"
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
using namespace std;
using namespace Eigen;
using namespace Spectra;

const int NUM_TRIALS = 1000;
const int SEED = 42;
const int CAP = 100 * 1000;
const int BUCKETS[] = {10, 100, 1000, 10000};

map<int, pair<double, double> > spectral_embeddings;
map<int, vector<double> > node2vec_embeddings;

void generateSpectralEmbeddings(PUNGraph& G) {
    map<int, int> nodeIdxToMatrixIdx;
    int index = 0;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        nodeIdxToMatrixIdx[NI.GetId()] = index++;

    int N = G->GetNodes();
    SparseMatrix<double> laplacian(N, N);
    laplacian.reserve(VectorXi::Constant(N, N + 2*G->GetEdges()));
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++) {
        int u = nodeIdxToMatrixIdx[NI.GetId()];
        laplacian.insert(u, u) = NI.GetOutDeg();
        for (int i = 0; i < NI.GetOutDeg(); i++) {
            int v = nodeIdxToMatrixIdx[NI.GetOutNId(i)];
            laplacian.insert(u, v) = -1; 
        }
    }
    
    SparseSymMatProd<double> op(laplacian);
    SymEigsSolver<double, SMALLEST_MAGN, SparseSymMatProd<double> > eigs(&op, 3, N/3);

    eigs.init();
    eigs.compute();

    if (eigs.info() != SUCCESSFUL) {
        cout << "Could not compute eigenvectors.\n";
        return;
    }
    
    for (map<int, int>::iterator it = nodeIdxToMatrixIdx.begin(); it != nodeIdxToMatrixIdx.end(); it++)
        spectral_embeddings[it->first] = make_pair(eigs.eigenvectors()(it->second, 1), eigs.eigenvectors()(it->second, 0));

    //cout << eigs.eigenvalues() << endl;
    //cout << eigs.eigenvectors() << endl;
}

void generateNode2vecEmbeddings(const string& filename) {
    string embedding_file = filename.substr(0, filename.size() - 4) + ".emb";
    ifstream fin(embedding_file);
    int N, D;
    fin >> N >> D;
    for (int i = 0, id; i < N; i++) {
        fin >> id;
        vector<double>& v = node2vec_embeddings[id];
        for (int j = 0; j < D; j++) {
            double val;
            fin >> val;
            v.push_back(val);
        }
    }
    fin.close();
}

// Returns a random neighbor, or -1 if there are none.
int randomNeighbor(TUNGraph::TNodeI NI) {
    if (NI.GetOutDeg() == 0)
        return -1;
    return NI.GetOutNId(rand() % NI.GetOutDeg());
}

double getSpectralDist(int a, int b) {
    pair<double, double> pa = spectral_embeddings[a];
    pair<double, double> pb = spectral_embeddings[b];
    return sqrt((pa.first - pb.first)*(pa.first - pb.first) + (pa.second - pb.second)*(pa.second - pb.second));
}


double getNode2VecDist(int a, int b) {
    double ret = 0.0;
    vector<double>& pa = node2vec_embeddings[a];
    vector<double>& pb = node2vec_embeddings[b];
    for (int i = 0; i < pa.size(); i++)
        ret += (pa[i] - pb[i])*(pa[i] - pb[i]);
    return sqrt(ret);
}

int getRandomNumber(int starting, int numberValues) {
    return rand() % (numberValues + 1) + starting;
}

int selectWeightedNodes(vector<int>& nodes, vector<int>& weights, int totalWeight) {

    int rnd = getRandomNumber(0, totalWeight);
    for (int i=0; i < weights.size(); i++) {
        if(rnd <= weights[i])
            return nodes[i];
        rnd -= weights[i];
    }

    return -1;
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


/* Returns the unvisited neighbor from a probability distribution
 * of nodes that are weighted by their degrees.
 * If all neighbors are visited, returns a random neighbor, or -1 if there are none
 */
int randomWeightedDegreeStrategy(PUNGraph& G, int cur, const set<int>& visited) {
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
        //int nxt = spectralStrategy(G, cur, dst, visited);
        //int nxt = node2vecStrategy(G, cur, dst, visited);
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

    spectral_embeddings.clear();
    //generateSpectralEmbeddings(G);
    node2vec_embeddings.clear();
    generateNode2vecEmbeddings(filename);

    vector<pair<int, int> > samples;
    getSamples(G, samples);

    cout << "Simulating degree strategy\n";
    simulate(G, samples, degreeStrategy);
    
    cout << "Simulating random unvisited strategy\n";
    simulate(G, samples, randomUnvisitedStrategy);

    cout << "Simulating random strategy\n";
    simulate(G, samples, randomStrategy);

    cout << "Simulating random degree weighting strategy\n";
    simulate(G, samples, randomWeightedDegreeStrategy);
    
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
    experiment("data/synthetic/powerlaw0.txt");
    experiment("data/synthetic/prefattach0.txt");
    
    experiment("data/synthetic/gnm_small0.txt");
    experiment("data/synthetic/smallworld_small0.txt");
    experiment("data/synthetic/powerlaw_small0.txt");
    experiment("data/synthetic/prefattach_small0.txt");

    return 0;
}
