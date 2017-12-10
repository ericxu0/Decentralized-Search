include ../../Makefile.config

TARGETS=search degree create_graphs spectral nodeEmbeddings gen_train_data
CXXFLAGSx = $(filter-out -std=c++98,$(CXXFLAGS))
CXXFLAGS2 = $(filter-out -DNDEBUG,$(CXXFLAGSx))
CXXFLAGS2 += -std=c++11 -Wno-delete-non-virtual-dtor
LIBS +=  -lboost_system -lboost_filesystem


all: $(TARGETS)

%: %.cpp
	$(CC) $(CXXFLAGS2) -o $@ $< $(DEPCPP) $(EXSNAP)/Snap.o -I$(EXSNAP) -I$(EXSNAPADV) -I$(EXGLIB) -I$(EXSNAPEXP) $(LDFLAGS) $(LIBS) 

clean:
	rm -f *.o $(TARGETS) *.Err
