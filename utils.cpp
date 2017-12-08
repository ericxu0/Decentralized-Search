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
#include <boost/filesystem.hpp>

using namespace std;
using namespace Eigen;
using namespace Spectra;
namespace fs = ::boost::filesystem;

map<int, pair<double, double> > spectral_embeddings;
map<int, vector<double> > node2vec_embeddings;
map<int, vector<int> > similarity_features;

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

void generateSpectralEmbeddings(PUNGraph& G, map<int, int>& compIdx) {
    int N = G->GetNodes();
    SparseMatrix<double> laplacian(N, N);
    laplacian.reserve(VectorXi::Constant(N, N + 2*G->GetEdges()));
    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++) {
        int u = compIdx[NI.GetId()];
        laplacian.insert(u, u) = NI.GetOutDeg();
        for (int i = 0; i < NI.GetOutDeg(); i++) {
            int v = compIdx[NI.GetOutNId(i)];
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
    
    for (auto &e : compIdx)
        spectral_embeddings[e.first] = make_pair(eigs.eigenvectors()(e.second, 1), eigs.eigenvectors()(e.second, 0));

    //cout << eigs.eigenvalues() << endl;
    //cout << eigs.eigenvectors() << endl;
}

void generateNode2vecEmbeddings(const string& filename) {
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

void generateSimilarityFeatures(const string& filename) {
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

double getSpectralDist(int a, int b) {
    pair<double, double> pa = spectral_embeddings[a];
    pair<double, double> pb = spectral_embeddings[b];
    return sqrt((pa.first - pb.first)*(pa.first - pb.first) + (pa.second - pb.second)*(pa.second - pb.second));
}

double getNode2VecDist(int a, int b) {
    double ret = 0.0;
    vector<double>& pa = node2vec_embeddings[a];
    vector<double>& pb = node2vec_embeddings[b];
    for (size_t i = 0; i < pa.size(); i++)
        ret += (pa[i] - pb[i])*(pa[i] - pb[i]);
    return sqrt(ret);
}

int getSimilarity(int a, int b) {
    int len = similarity_features[a].size();
    assert(similarity_features[a].size() == similarity_features[b].size());
    int cnt = 0;
    for (int i = 0; i < len; i++)
        cnt += similarity_features[a][i] == 1 && similarity_features[b][i] == 1;
    return cnt;
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
