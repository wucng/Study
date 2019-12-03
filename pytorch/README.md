- https://github.com/pytorch/examples/tree/master/cpp
- https://www.pytorchtutorial.com/tag/pytorch-c-api/
- https://pytorch.org/
- https://pytorch.org/cppdocs/
- https://pytorch.org/tutorials/advanced/cpp_frontend.html
- https://github.com/ShigekiKarita/thxx


---
# 0.环境
```c
Ubuntu 16.04
gcc-5.4.0
g++-5.4.0
cmake 3.5.2
NVIDIA GTX 1060 6G
NVIDIA-418.87.01
cuda 10.0
cudnn 7.6
```
# 1.安装
```c
// libTorch1.3.0-cu100 c++11
wget https://download.pytorch.org/libtorch/cu100/libtorch-cxx11-abi-shared-with-deps-1.3.0.zip
// libTorch1.3.1-cu101 c++11
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip

unzip libtorch-cxx11-abi-shared-with-deps-1.3.0.zip -d /opt
```
# 2.测试
## `dcgan.cpp`
```c
// dcgan.cpp

#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
}
```
## `CMakeLists.txt`
方式一：
```python
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

# 指定libTorch位置，加上这个 执行cmake时可以不用加上 -DCMAKE_PREFIX_PATH=/opt/libtorch
set(Torch_DIR /opt/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)
```
方式二：
```c
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)
set(CMAKE_CXX_STANDARD 11)
set(Torch_DIR /opt/libtorch/share/cmake/Torch) #指定libTorch位置(应该是有更好的办法安装)

#include_directories(${OpenCV_INCLUDE_DIRS} /opt/libtorch/include /opt/libtorch/include/torch/csrc/api/include)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)    # 自动查找libTorch包

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan ${OpenCV_LIBS}  ${TORCH_LIBRARIES}) # 加入libTorch的库文件路径
```



## 目录结构
```c
dcgan/
  CMakeLists.txt
  dcgan.cpp
```
## 编译
```c
mkdir build
cd build
// CMakeLists.txt 使用方式一
cmake [-DCMAKE_PREFIX_PATH=/opt/libtorch] ..
make -j4

// CMakeLists.txt 使用方式二
cmake ..
make -j4
```

# 3.定义神经网络模型
```c
// python: torch.nn.Module
// c++: torch::nn::Module
```
`python方式`：
```python
import torch

class Net(torch.nn.Module):
  def __init__(self, N, M):
    super(Net, self).__init__()
    self.W = torch.nn.Parameter(torch.randn(N, M))
    self.b = torch.nn.Parameter(torch.randn(M))

  def forward(self, input):
    return torch.addmm(self.b, input, self.W)
```
`C++方式`
```c
#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) {
    W = register_parameter("W", torch::randn({N, M}));
    b = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return torch::addmm(b, input, W);
  }
  torch::Tensor W, b;
};
```
---
`python`
```python
class Net(torch.nn.Module):
  def __init__(self, N, M):
      super(Net, self).__init__()
      # Registered as a submodule behind the scenes
      self.linear = torch.nn.Linear(N, M)
      self.another_bias = torch.nn.Parameter(torch.rand(M))

  def forward(self, input):
    return self.linear(input) + self.another_bias
 

>>> net = Net(4, 5)
>>> print(list(net.parameters()))
```
`c++`
```c
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
      : linear(register_module("linear", torch::nn::Linear(N, M))) {
    another_bias = register_parameter("b", torch::randn(M));
  }
  torch::Tensor forward(torch::Tensor input) {
    return linear(input) + another_bias;
  }
  torch::nn::Linear linear;
  torch::Tensor another_bias;
};
```
You can find the full list of available built-in modules like `torch::nn::Linear`, `torch::nn::Dropout` or `torch::nn::Conv2d` in the documentation of the torch::nn namespace [here](https://pytorch.org/cppdocs/api/namespace_torch__nn.html).

Calling `parameters()` returns a `std::vector<torch::Tensor>`, which we can iterate over:
```c
int main() {
  Net net(4, 5);
  for (const auto& p : net.parameters()) {
    std::cout << p << std::endl;
  }
}
```
具有三个参数，就像在Python中一样。 为了也查看这些参数的名称，C++ API提供了`named_parameters()`方法，该方法返回的`OrderedDict`就像在Python中一样：
```c
Net net(4, 5);
for (const auto& pair : net.named_parameters()) {
  std::cout << pair.key() << ": " << pair.value() << std::endl;
}
```
要使用C ++执行网络，我们只需调用自己定义的`forward()`方法即可：
```c
int main() {
  Net net(4, 5);
  std::cout << net.forward(torch::ones({2, 4})) << std::endl;
}
```


```c
struct Net : torch::nn::Module { };

void a(Net net) { }
void b(Net& net) { }
void c(Net* net) { }

int main() {
  Net net;
  a(net);
  a(std::move(net));
  b(net);
  c(&net);
}
```
对于第二种情况-参考语义-我们可以使用`std::shared_ptr`。引用语义的优点是，像在Python中一样，它减少了思考模块如何传递给函数以及如何声明参数（假设您`shared_ptr`在任何地方使用）的认知开销。
```c
struct Net : torch::nn::Module {};

void a(std::shared_ptr<Net> net) { }

int main() {
  auto net = std::make_shared<Net>();
  a(net);
}
```

```c
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M)
    : linear(register_module("linear", torch::nn::Linear(N, M)))
  { }
  torch::nn::Linear linear;
};
```
或者
```c
struct Net : torch::nn::Module {
  Net(int64_t N, int64_t M) {
    linear = register_module("linear", torch::nn::Linear(N, M));
  }
  torch::nn::Linear linear{nullptr}; // construct an empty holder
};
```
