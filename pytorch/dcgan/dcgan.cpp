// https://github.com/pytorch/examples/blob/master/cpp/dcgan/dcgan.cpp
#include <torch/torch.h>

#include <cmath>
#include <cstdio>
#include <iostream>

using namespace torch;

const int kBatchSize=64;
const int kNoiseSize=100;

int main()
{
  nn::Sequential generator(
    // Layer 1
    nn::Conv2d(nn::Conv2dOptions(kNoiseSize, 256, 4)
                   .with_bias(false)
                   .transposed(true)),
    nn::BatchNorm(256),
    nn::Functional(torch::relu),
    // Layer 2
    nn::Conv2d(nn::Conv2dOptions(256, 128, 3)
                   .stride(2)
                   .padding(1)
                   .with_bias(false)
                   .transposed(true)),
    nn::BatchNorm(128),
    nn::Functional(torch::relu),
    // Layer 3
    nn::Conv2d(nn::Conv2dOptions(128, 64, 4)
                   .stride(2)
                   .padding(1)
                   .with_bias(false)
                   .transposed(true)),
    nn::BatchNorm(64),
    nn::Functional(torch::relu),
    // Layer 4
    nn::Conv2d(nn::Conv2dOptions(64, 1, 4)
                   .stride(2)
                   .padding(1)
                   .with_bias(false)
                   .transposed(true)),
    nn::Functional(torch::tanh));


  nn::Sequential discriminator(
    // Layer 1
    nn::Conv2d(
        nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).with_bias(false)),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 2
    nn::Conv2d(
        nn::Conv2dOptions(64, 128, 4).stride(2).padding(1).with_bias(false)),
    nn::BatchNorm(128),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 3
    nn::Conv2d(
        nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).with_bias(false)),
    nn::BatchNorm(256),
    nn::Functional(torch::leaky_relu, 0.2),
    // Layer 4
    nn::Conv2d(
        nn::Conv2dOptions(256, 1, 3).stride(1).padding(0).with_bias(false)),
    nn::Functional(torch::sigmoid));

  // 加载数据
  auto dataset = torch::data::datasets::MNIST("/media/wucong/work/practice/work/datas/mnist")
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      torch::data::DataLoaderOptions().batch_size(kBatchSize).workers(2));

  for (torch::data::Example<>& batch : *data_loader) {
    std::cout << "Batch size: " << batch.data.size(0) << " | Labels: ";
    for (int64_t i = 0; i < batch.data.size(0); ++i) {
      std::cout << batch.target[i].item<int64_t>() << " ";
    }
    std::cout << std::endl;
  }
}
