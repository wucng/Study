all:
	g++ 01_load_image.cpp `pkg-config --cflags --libs opencv` -o a.out
test:
	./a.out ../data/test.jpg
clean:
	rm -rf a.out
