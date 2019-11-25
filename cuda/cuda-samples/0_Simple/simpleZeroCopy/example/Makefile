all:
	nvcc function.cu -I ../../../common/ -std=c++11
	# nvcc main.cu -I ../../../common/ -I ./ -std=c++11 # 报错
run:
	./a.out
clean:
	rm -rf a.out
