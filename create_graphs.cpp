#include "Snap.h"
#include <sstream>

using namespace std;

const int SEED = 42;
const int NUM = 3;

const int V = 10000;       // Vertices
const int E = 100000;      // Edges for Gnm
const int Vsmall = 1000;       // Vertices
const int Esmall = 10000;      // Edges for Gnm
const int EXP = 2;         // Power law exponent
const int OUT_DEG = 10;    // Out-degree for Small World and Preferential Attachment
const double REWIRE = 0.5; // Rewire probability for Small World

const string FOLDER = "data/synthetic/";

PUNGraph gnm(TRnd& rnd, bool small) {
    return TSnap::GenRndGnm<PUNGraph>(small ? Vsmall : V, small ? Esmall : E, false, rnd);
}
PUNGraph powerlaw(TRnd& rnd, bool small) {
    return TSnap::GenRndPowerLaw(small ? Vsmall : V, EXP, false, rnd);
}
PUNGraph smallworld(TRnd& rnd, bool small) {
    return TSnap::GenSmallWorld(small ? Vsmall : V, OUT_DEG, REWIRE, rnd);
}
PUNGraph prefattach(TRnd& rnd, bool small) {
    return TSnap::GenPrefAttach(small ? Vsmall : V, OUT_DEG);
}

void create(const string& name, bool small, PUNGraph (*newGraph)(TRnd&, bool)) {
    TRnd rnd;
    rnd.PutSeed(SEED);
    for (int i = 0; i < NUM; i++) {
        PUNGraph graph = newGraph(rnd, small);
        stringstream ss;
        ss << FOLDER << name << i << ".txt";
        TSnap::SaveEdgeList(graph, ss.str().c_str());
    }
}

int main() {
    create("gnm", 0, gnm);
    //create("powerlaw", 0, powerlaw); // This doesn't work for some reason?
    create("smallworld", 0, smallworld);
    create("prefattach", 0, prefattach);
    create("gnm_small", 1, gnm);
    //create("powerlaw_small", 1, powerlaw); // This doesn't work for some reason?
    create("smallworld_small", 1, smallworld);
    create("prefattach_small", 1, prefattach);

    return 0;
}
