# 1.测试
```c
$ g++ main.cc cSample.cc -std=c++11

$ ./a.out
```

# 2.编译成动态库
```c
// 生成cSample.o
g++ cSample.cc -c -fpic -std=c++11 

// 生成libcSample.so(动态库文件) 格式为libxxxx.so 只能更改xxxx
g++ -shared cSample.o -o libcSample.so -std=c++11 
```

# 3.C++调用动态库文件
```c
// -l cSample  加上动态编译库 libcSample.so ,[..]可选
$ g++ main.cc [-I ./ -L ./] -l cSample -std=c++11

$ ./a.out
```
# 4.python调用动态库文件
## 4.1 使用`ctypes`
参考:[python调用.so](https://www.cnblogs.com/fariver/p/6573112.html) 使用`ctypes`
```c
$ g++ cSample2.cc -c -fpic
$ g++ -shared cSample2.o -o libcSample.so
// or g++ cSample2.cc -fPIC -shared -o libcSample.so
$ python3 pyCallSo.py

// 使用nvcc编译需加上 -Xcompiler -fPIC
$ nvcc cSample2.cc -Xcompiler -fPIC -shared -o libcSample.so
```

==拓展：调用GPU==
```c
$ nvcc cSample2.cu -Xcompiler -fPIC -shared -o libcSample.so
$ python3 pyCallSo.py
```

## 4.2 使用`cython`(推荐)
```c
$ cython -2 pycSample.pyx -o pycSample.cc
$ g++ -g -O2 -fpic -c pycSample.cc -o pycSample.o `python3-config --includes`
$ g++ -g -O2 -fpic -c cSample.cc -o cSample.o
$ g++ -g -O2 -shared -o pycSample.so cSample.o pycSample.o `python3-config --libs`

$ python3 pyCallSo_cython.py
$ rm -rf *.so *.o pycSample.cc
```

==拓展：调用GPU==
```c
$ cython -2 pycSample.pyx -o pycSample.cu
$ nvcc -g -O2 -Xcompiler -fpic -c pycSample.cu -o pycSample.o `python3-config --includes`
$ nvcc -g -O2 -Xcompiler -fpic -c cSample.cu -o cSample.o
$ nvcc -g -O2 -shared -o pycSample.so cSample.o pycSample.o `python3-config --libs`

$ python3 pyCallSo_cython.py
$ rm -rf *.so *.o pycSample.cu
```