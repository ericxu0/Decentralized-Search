include ../../Makefile.config

TARGETS=search degree

all: $(TARGETS)

%: %.cpp
	$(CC) $(CXXFLAGS) -o $@ $< $(DEPCPP) $(EXSNAP)/Snap.o -I$(EXSNAP) -I$(EXSNAPADV) -I$(EXGLIB) -I$(EXSNAPEXP) $(LDFLAGS) $(LIBS)

clean:
	rm -f *.o $(TARGETS)
