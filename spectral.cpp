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

void outputEmbeddings(const string& filename) {
    PUNGraph G = TSnap::LoadEdgeList<PUNGraph>(filename.c_str(), 0, 1);
    G = TSnap::GetMxWcc(G);
    
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
    int nconv = eigs.compute();
    if (eigs.info() != SUCCESSFUL) {
        cout << "Could not compute eigenvectors.\n";
        return;
    }
    
    string output_file = filename.substr(0, filename.size() - 4) + ".out";
    ofstream fout(output_file);
    fout << N << "\n";
    for (map<int, int>::iterator it = nodeIdxToMatrixIdx.begin(); it != nodeIdxToMatrixIdx.end(); it++)
        fout << fixed << setprecision(10) << it->first << " " << eigs.eigenvectors()(it->second, 1) << " " << eigs.eigenvectors()(it->second, 0) << "\n";

    fout.close();
}

int main() {

    outputEmbeddings("data/synthetic/gnm_small0.txt");
    outputEmbeddings("data/synthetic/smallworld_small0.txt");
    outputEmbeddings("data/synthetic/powerlaw_small0.txt");
    outputEmbeddings("data/synthetic/prefattach_small0.txt");
    
    return 0;
}
