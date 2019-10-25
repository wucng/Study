all:
	g++ 16_drawByPixel.cpp `pkg-config --cflags --libs opencv` -o a.out
test:
	./a.out ../data/test.jpg
clean:
	rm -rf a.out
