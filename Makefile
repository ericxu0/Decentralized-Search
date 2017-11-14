include ../../Makefile.config

TARGETS=search degree create_graphs spectral nodeEmbeddings
CXXFLAGS2 = $(filter-out -std=c++98,$(CXXFLAGS))
CXXFLAGS2 += -std=c++11

all: $(TARGETS)

%: %.cpp
	$(CC) $(CXXFLAGS2) -o $@ $< $(DEPCPP) $(EXSNAP)/Snap.o -I$(EXSNAP) -I$(EXSNAPADV) -I$(EXGLIB) -I$(EXSNAPEXP) $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o $(TARGETS) *.Err
