CXXFLAGS+=-Iinclude -std=c++11 -Wall `pkg-config --cflags adolc`
LDFLAGS+= -larmadillo `pkg-config --libs adolc`
all: TestConjGrad TestMnist

TestConjGrad: src/Optimizable.o test/testconjgrad.o
	$(CXX) $(LDFLAGS) -o TestConjGrad src/Optimizable.o test/testconjgrad.o
TestMnist: src/Optimizable.o test/testmnist.o
	$(CXX) $(LDFLAGS) -o TestMnist src/Optimizable.o test/testmnist.o
clean:
	rm -f TestConjGrad TestMnist test/*.o src/*.o
%.o: %.cxx
	$(CXX) $(CXXFLAGS) %(CPPFLAGS) -c $<
