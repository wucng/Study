all:
	cython -2 pycSample.pyx -o pycSample.cc
	g++ -g -O2 -fpic -c pycSample.cc -o pycSample.o `python3-config --includes`
	g++ -g -O2 -fpic -c cSample.cc -o cSample.o
	g++ -g -O2 -shared -o pycSample.so cSample.o pycSample.o `python3-config --libs`

clean:
	rm -rf *.so *.o pycSample.cc
