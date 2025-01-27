CXX      = g++ -fPIC -Dlearn_fft_scalar=double -Dkiss_fft_scalar=double -mavx
CXXFLAGS = -Wall -ansi -std=c++17 -O2 -Ikissfft/ -I../src
CC       = gcc -fPIC -O2  -Dlearn_fft_scalar=double -Dkiss_fft_scalar=double  -Ikissfft/  -I../src  -mavx
CCFLAGS  =
LDFLAGS  =
LIBS     =
OBJS     = 
MAINOBJ  = test_fft.o kissfft/kiss_fft.o kissfft/kiss_fftr.o pffft/pffft.o

OBJ/%.o : %.cpp %.cc
	$(CXX) -c $(CXXFLAGS) $<

OBJ/%.o : %.c
	$(CC) -c $(CCFLAGS) $<

test_fft : $(MAINOBJ) $(OBJS)
	$(CXX) -o $@ $(MAINOBJ) $(OBJS)  $(LDFLAGS) $(LIBS)
	
exec : test_fft

debug: CXXFLAGS += -DDEBUG -g
debug: CCFLAGS += -DDEBUG -g
debug: exec

clean:
	rm -f $(MAINOBJ) $(OBJS)