#include "Snap.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = ::boost::filesystem;

const int INFTY = 1<<28;

double AVG_PATH_LEN;
map<int, double> clusterCf;
map<int, vector<double> > node2vec_embeddings;
map<int, vector<int> > similarity_features;
map<int, map<int, double> > tfidf;

vector<double> ridgeWeights;
vector<double> overallWeights;
map<int, map<int, double> > similarityCache;

const int CHUNKS = 100;
double discretizedQ[CHUNKS + 1];

PUNGraph loadGraph(string filename) {
    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>(filename.c_str(), 0, 1);
    cout << "# Nodes: " << G->GetNodes() << endl;
    cout << "# Edges: " << G->GetEdges() << endl;
    G = TSnap::GetMxWcc(G);
    TSnap::DelSelfEdges(G);
    cout << "# Nodes (Max WCC): " << G->GetNodes() << endl;
    cout << "# Edges (Max WCC): " << G->GetEdges() << endl;
    cout << endl;
    return G;
}

void computeCompressionIndices(PUNGraph G, map<int, int>& compIdx) {
    int index = 0;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        compIdx[NI.GetId()] = index++;
}

void computeClusterCf(PUNGraph G) {
    clusterCf.clear();
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        clusterCf[NI.GetId()] = TSnap::GetNodeClustCf(G, NI.GetId());
}

void computeShortestPaths(PUNGraph& G, map<int, int>& compIdx, vector<vector<int> >& minDist) {
    int N = G->GetNodes();
    for (int i = 0; i < N; i++)
        minDist.push_back(vector<int>(N, INFTY));
    for (int i = 0; i < N; i++)
        minDist[i][i] = 0;
    for (TUNGraph::TEdgeI EI = G->BegEI(); EI < G->EndEI(); EI++) {
        int a = compIdx[EI.GetSrcNId()];
        int b = compIdx[EI.GetDstNId()];
        minDist[a][b] = minDist[b][a] = 1; // undirected
    }
    for (int k = 0; k < N; k++)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                minDist[i][j] = min(minDist[i][j], minDist[i][k] + minDist[k][j]);

    double avg = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++)
            avg += minDist[i][j];
    AVG_PATH_LEN = avg/(N*(N - 1)/2);
}

void getSamples(PUNGraph& G, vector<pair<int, int> >& samples, int numSamples) {
    vector<int> nodes;
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++)
        nodes.push_back(NI.GetId());
    int N = nodes.size();

    for (int i = 0; i < numSamples; ) {
        int src = nodes[rand() % N];
        int dst = nodes[rand() % N];
        if (src != dst) {
            samples.push_back(make_pair(src, dst));
            i++;
        }
    }
}

string getBase(string s) {
    int idx = s.size() - 1;
    while (idx >= 0 && s[idx] != '.')
        idx--;
    return s.substr(0, idx);
}


// return the filenames of all files that have the specified extension
// in the specified directory and all subdirectories
vector<string> getAllFiles(const fs::path& root, const string& ext)
{
    vector<string> ret;

    if(!fs::exists(root) || !fs::is_directory(root)) return ret;

    fs::recursive_directory_iterator it(root);
    fs::recursive_directory_iterator endit;

    while(it != endit)
    {
        if(fs::is_regular_file(*it) && it->path().extension() == ext) {
            ret.push_back(it->path().filename().string());
        }
        ++it;

    }

    return ret;

}

void generateNode2vecEmbeddings(const string& filename) {
    node2vec_embeddings.clear();
    string embedding_file = getBase(filename) + ".emb";
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

void generateSimilarityFeatures(const string& filename, bool isCitation) {
    if (isCitation) {
        tfidf.clear();
        ifstream fin(getBase(filename) + ".tfidf");
        string input;
        while (getline(fin, input) && input != "") {
            stringstream ss(input);
            int x, word;
            double value;
            char c;
            ss >> c >> x >> c >> word >> c >> c >> value;
            tfidf[x][word] = value;
        }
    } else {
        similarity_features.clear();
        string feat_file = getBase(filename) + ".feat";
        ifstream fin(feat_file);
        string input;
        while (getline(fin, input) && input != "") {
            stringstream ss(input);
            int x;
            ss >> x;
            vector<int>& v = similarity_features[x];
            while (ss >> x) {
                v.push_back(x);
            }
        }
        fin.close();
    }
}

void getRegressionWeights(const string& filename, bool isCitation, bool useNode2Vec) {
    ridgeWeights.clear();

    string base = getBase(filename);
    string suffix = useNode2Vec ? "_n2v" : "_sim";
    int idx = base.size() - 1;
    while (idx >= 0 && base[idx] != '/')
        idx--;
    string num = base.substr(idx + 1, base.size() - idx - 1);
    string path;
    if (isCitation) {
        path = "data/training_data/Ridge_cit-HepTh-subset" + suffix + ".weights";
    } else {
        path = "data/training_data/Ridge_facebook-" + num + suffix + ".weights";
    }
    ifstream fin(path);
    double x;
    while (fin >> x)
        ridgeWeights.push_back(x);
    fin.close();

    overallWeights.clear();
    if (useNode2Vec)
        path = "data/training_data/Ridge_all_n2v.weights";
    else
        path = "data/training_data/Ridge_all_sim.weights";
    ifstream fin2(path);
    while (fin2 >> x)
        overallWeights.push_back(x);
    fin2.close();
}

double getNode2VecL2Dist(int a, int b) {
    double ret = 0.0;
    vector<double>& pa = node2vec_embeddings[a];
    vector<double>& pb = node2vec_embeddings[b];
    for (size_t i = 0; i < pa.size(); i++)
        ret += (pa[i] - pb[i])*(pa[i] - pb[i]);
    return sqrt(ret);
}

double cosineSimilarity(const vector<int>& a, const vector<int>& b) {
    double dot = 0.0;
    double lenA = 0.0;
    double lenB = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        lenA += a[i] * a[i];
        lenB += b[i] * b[i];
    }
    return dot / sqrt(lenA * lenB);
}

double cosineSimilarity(const vector<double>& a, const vector<double>& b) {
    double dot = 0.0;
    double lenA = 0.0;
    double lenB = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        lenA += a[i] * a[i];
        lenB += b[i] * b[i];
    }
    return dot / sqrt(lenA * lenB);
}

double cosineSimilarity(map<int, double>& a, map<int, double>& b) {
    double dot = 0.0;
    double lenA = 0.0;

    for (auto it = a.begin(); it != a.end(); it++) {
        int index = it->first;
        double valA = it->second;
        double valB = b[index];
        dot += valA * valB;
        lenA += valA * valA;
    }
    double lenB = 0.0;
    for (auto it = b.begin(); it != b.end(); it++)
        lenB += it->second * it->second;
    return dot / sqrt(lenA * lenB);
}

const int SIMILARITY_TFIDF = 0;
const int SIMILARITY_FEATURES = 1;
const int SIMILARITY_NODE2VEC = 2;

double getSimilarity(int a, int b, int similarityMethod) {
    if (b < a) {
        swap(a, b);
    }

    auto it = similarityCache.find(a);
    if (it != similarityCache.end()) {
        map<int, double>& m = it->second;
        auto it2 = m.find(b);
        if (it2 != m.end())
            return it2->second;
    }

    double result = 0.0;
    if (similarityMethod == SIMILARITY_TFIDF) {
        result = cosineSimilarity(tfidf[a], tfidf[b]);
    } else if (similarityMethod == SIMILARITY_FEATURES) {
        result = cosineSimilarity(similarity_features[a], similarity_features[b]);
    } else if (similarityMethod == SIMILARITY_NODE2VEC) {
        result = 0.0 - getNode2VecL2Dist(a, b);
    } else {
        cout << "Unknown similarity method: " << similarityMethod << endl;
        assert(false);
    }

    similarityCache[a][b] = similarityCache[b][a] = result;
    return result;
}

void normalizeSimilarity(PUNGraph& G, int similarityMethod) {
    similarityCache.clear();
    double maxSim = 0;
    double minSim = 1E23;
    for (TUNGraph::TNodeI NI1 = G->BegNI(); NI1 < G->EndNI(); NI1++) {
        for (TUNGraph::TNodeI NI2 = G->BegNI(); NI2 < G->EndNI(); NI2++) {
            int a = NI1.GetId();
            int b = NI2.GetId();
            if (a != b) {
                double sim = getSimilarity(a, b, similarityMethod);
                maxSim = max(maxSim, sim);
                minSim = min(minSim, sim);
            }
        }
    }

    for (TUNGraph::TNodeI NI1 = G->BegNI(); NI1 < G->EndNI(); NI1++) {
        for (TUNGraph::TNodeI NI2 = G->BegNI(); NI2 < G->EndNI(); NI2++) {
            int a = NI1.GetId();
            int b = NI2.GetId();
            similarityCache[a][b] = (similarityCache[a][b] - minSim) / (maxSim - minSim);
            assert(similarityCache[a][b] >= 0.0 && similarityCache[a][b] <= 1.0);
        }
    }
}

void computeDiscretizedQ(PUNGraph& G) {
    // compute all pairwise similarity
    int counts[CHUNKS + 1];
    fill(counts, counts + CHUNKS + 1, 0);
    fill(discretizedQ, discretizedQ + CHUNKS + 1, 0);
    for (TUNGraph::TNodeI NI1 = G->BegNI(); NI1 < G->EndNI(); NI1++) {
        for (TUNGraph::TNodeI NI2 = G->BegNI(); NI2 < G->EndNI(); NI2++) {
            int a = NI1.GetId();
            int b = NI2.GetId();
            double sim = similarityCache[a][b];
            assert(sim >= 0.0 && sim <= 1.0);
            int index = (int) (sim * CHUNKS);
            if (G->IsEdge(a, b))
                discretizedQ[index] += 1;
            counts[index]++;
        }
    }
    for (int i = 0; i <= CHUNKS; i++) {
        if (counts[i] > 0)
            discretizedQ[i] /= counts[i];
        else
            discretizedQ[i] = -1.0;
    }
}

double qst(double sim) {
    return discretizedQ[(int) (sim * CHUNKS)];
}

// requires calling computeDiscretizedQ() beforehand, which requires calling normalizeSimilarity() before that
double pst(PUNGraph& G, int src, int target) {
    return 1.0 - pow(1.0 - qst(similarityCache[src][target]), G->GetNI(src).GetOutDeg());
}

void getFeatureVector(PUNGraph& G, int cur, int dst, const set<int>& visited, vector<double>& feat) {
    TUNGraph::TNodeI curNI = G->GetNI(cur);
    int visitedNeighbors = 0;
    for (int i = 0; i < curNI.GetOutDeg(); i++) {
        int x = curNI.GetOutNId(i);
        visitedNeighbors += visited.find(x) != visited.end();
    }
    double fracVisited = 1.0 * visitedNeighbors / curNI.GetOutDeg();

    feat.push_back(AVG_PATH_LEN);                        // 0. average shortest path length in G
    feat.push_back(similarityCache[cur][dst]);           // 1. similarity OR L2 distance of node2vec
    feat.push_back(curNI.GetOutDeg());                   // 2. degree
    feat.push_back(clusterCf[cur]);                      // 3. clustering coefficient
    feat.push_back(visited.find(cur) == visited.end());  // 4. 1 if unvisited, 0 if visited
    feat.push_back(fracVisited);                         // 5. fraction of visited neighbors
    feat.push_back(pst(G, cur, dst));                    // 6. EVN probability
    feat.push_back(1);                                   // 7. constant term for linear regression
}

// Returns a random neighbor, or -1 if there are none.
int randomNeighbor(TUNGraph::TNodeI NI) {
    if (NI.GetOutDeg() == 0)
        return -1;
    return NI.GetOutNId(rand() % NI.GetOutDeg());
}

int getRandomNumber(int starting, int numberValues) {
    return rand() % (numberValues + 1) + starting;
}

int randomElement(vector<int> choices) {
    return choices[rand() % choices.size()];
}

int selectWeightedNodes(vector<int>& nodes, vector<int>& weights, int totalWeight) {
    int rnd = getRandomNumber(0, totalWeight);
    for (size_t i=0; i < weights.size(); i++) {
        if(rnd <= weights[i])
            return nodes[i];
        rnd -= weights[i];
    }
    return -1;
}