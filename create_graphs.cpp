#include "Snap.h"
#include <sstream>

using namespace std;

const int SEED = 42;
const int NUM = 3;

const int V = 10000;       // Vertices
const int E = 100000;      // Edges for Gnm
const int EXP = 2;         // Power law exponent
const int OUT_DEG = 10;    // Out-degree for Small World and Preferential Attachment
const double REWIRE = 0.5; // Rewire probability for Small World

const string FOLDER = "data/synthetic/";

PUNGraph gnm(TRnd& rnd) {
    return TSnap::GenRndGnm<PUNGraph>(V, E, false, rnd);
}
PUNGraph powerlaw(TRnd& rnd) {
    return TSnap::GenRndPowerLaw(V, EXP, false, rnd);
}
PUNGraph smallworld(TRnd& rnd) {
    return TSnap::GenSmallWorld(V, OUT_DEG, REWIRE, rnd);
}
PUNGraph prefattach(TRnd& rnd) {
    return TSnap::GenPrefAttach(V, OUT_DEG);
}

void create(const string& name, PUNGraph (*newGraph)(TRnd&)) {
    TRnd rnd;
    rnd.PutSeed(SEED);
    for (int i = 0; i < NUM; i++) {
        PUNGraph graph = newGraph(rnd);
        stringstream ss;
        ss << FOLDER << name << i << ".txt";
        TSnap::SaveEdgeList(graph, ss.str().c_str());
    }
}

int main() {
    create("gnm", gnm);
    //create("powerlaw", powerlaw); // This doesn't work for some reason?
    create("smallworld", smallworld);
    create("prefattach", prefattach);

    return 0;
}
