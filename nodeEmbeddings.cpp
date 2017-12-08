#include "Snap.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <vector>
using namespace std;

map<int, vector<double> > node2vec_embeddings;

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

void visualizeNodeEmbeddings(const string& filename) {
    cout << "Visualizing Node Embeddings on " << filename << endl;
    node2vec_embeddings.clear();
    generateNode2vecEmbeddings(filename);

    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>(filename.c_str(), 0, 1);
    map<int,vector<int>, greater<int> > degreeMaps;


    for (TUNGraph::TNodeI NI = G->BegNI(); NI < G->EndNI(); NI++) {
        if ( degreeMaps.find(NI.GetOutDeg()) != degreeMaps.end() ) {
            degreeMaps[NI.GetOutDeg()].push_back(NI.GetId());
        }
        else {
            vector<int> nodes;
            nodes.push_back(NI.GetId());
            degreeMaps.insert({NI.GetOutDeg(), nodes});
        }
    }

    vector<int> topTenNodes;

    for (const auto& p : degreeMaps) {
        for (const auto& n : p.second) {
            topTenNodes.push_back(n);
            if(topTenNodes.size() == 10)
                break;
        }
        if(topTenNodes.size() == 10)
            break;
    }

    for (const auto& n : topTenNodes) {
        for (const auto& embedding: node2vec_embeddings[n]) {
            cout << embedding << " ";
        }
        cout << endl;
    }
        
}

int main() {

    visualizeNodeEmbeddings("data/real/facebook_combined.txt");


    return 0;
}
