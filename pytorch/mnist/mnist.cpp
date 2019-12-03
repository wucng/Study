/*
Net model; // 使用这个没法保存模型参数
model.to(device);
  
std::shared_ptr<Net> model = std::make_shared<Net>(); // 使用指针可以保存模型参数
model->to(device);

1.实现了pytorch c++ 训练mnist数据，数据下载：http://yann.lecun.com/exdb/mnist/
2.下载手动解压存到 xxxx/xx/mnist
3.模型保存与加载
*/

#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>


// Where to find the MNIST dataset.
const char* kDataRoot = "/media/wucong/work/practice/work/data/mnist";

// Set to `true` to restore models and optimizers from previously saved
// checkpoints.
const bool kRestoreFromCheckpoint = true;  //false;

// After how many batches to log a new update with the loss value.
const int64_t kLogInterval = 10;

// The batch size for training.
const int64_t kTrainBatchSize = 64;

// The batch size for testing.
const int64_t kTestBatchSize = 1000;

// The number of epochs to train.
const int64_t kNumberOfEpochs = 1;//10;

// ---------------------model----------------------
struct Net2:torch::nn::Module
{
    //构造函数初始化变量
    Net2():conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
        conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
        batch_norm(20),
        fc1(320, 50),
        fc2(50, 10) 
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("batch_norm", batch_norm);
        register_module("conv2_drop", conv2_drop);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    // 实现forward
    torch::Tensor forward(torch::Tensor x) 
    {
        x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
        x = torch::relu(
            torch::max_pool2d(conv2_drop->forward(conv2->forward(x)), 2));
        x = batch_norm->forward(x);
        x = x.view({-1, 320});
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, /*dim=*/1);
    }

    // 声明变量
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::BatchNorm batch_norm;
    torch::nn::FeatureDropout conv2_drop;
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
};


struct Net: public torch::nn::Module{
  Net(){
    conv1=register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, 3).stride(1).padding(1).with_bias(false)));
    conv2=register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, 3).stride(1).padding(1).with_bias(false)));
    // batch_norm=register_module("batch_norm", torch::nn::BatchNorm(20));
    batch_norm=register_module("batch_norm", torch::nn::BatchNorm(torch::nn::BatchNormOptions(20)));

    fc1 = register_module("fc1", torch::nn::Linear(980, 50));
    fc2 = register_module("fc2", torch::nn::Linear(50, 10));
  }

  // Implement Algorithm
  torch::Tensor forward(torch::Tensor x) 
  {
      x = torch::relu(torch::max_pool2d(conv1->forward(x), 2));
      x = torch::relu(
          torch::max_pool2d(conv2->forward(x), 2));
      x = batch_norm->forward(x);
      x = x.view({-1, 980});
      x = torch::relu(fc1->forward(x));
      x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
      x = fc2->forward(x);
      return torch::log_softmax(x, /*dim=*/1);
  }


  // Declare layers
  torch::nn::Conv2d conv1{nullptr},conv2{nullptr};
  torch::nn::BatchNorm batch_norm{nullptr};
  torch::nn::Linear fc1{nullptr},fc2{nullptr};
};

// 方式二
// torch::nn::Sequential model(
//   torch::nn::Conv2d(nn::Conv2dOptions(1,10,5)),
//   torch::nn::BatchNorm(10),
//   torch::nn::Functional(torch::relu),
//   torch::nn::MaxPool2d(2),

//   torch::nn::Conv2d(nn::Conv2dOptions(10,20,5)),
//   torch::nn::BatchNorm(20),
//   torch::nn::Functional(torch::relu),
//   torch::nn::MaxPool2d(2),
//   torch::nn::Dropout(0.2),

//   //torch::nn::Functional(torch::flatten),
//   //torch::nn::Flatten(),

//   torch::nn::Linear(320,50),
//   torch::nn::Functional(torch::relu),
//   torch::nn::Linear(50,10)
//   // torch::nn::Functional(torch::log_softmax(1))
// );


// ----------------训练---------------------------------
template <typename DataLoader>
void train(
    size_t epoch,
    std::shared_ptr<Net> model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::Optimizer& optimizer,
    size_t dataset_size) {
  model->train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model->forward(data);
    auto loss = torch::nll_loss(output, targets);
    AT_ASSERT(!std::isnan(loss.template item<float>()));
    loss.backward();
    optimizer.step();

    if (batch_idx++ % kLogInterval == 0) {
      std::printf(
          "\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
          epoch,
          batch_idx * batch.data.size(0),
          dataset_size,
          loss.template item<float>());
    }

  }
}

// ------------------------测试--------------------------
template <typename DataLoader>
void test(
    std::shared_ptr<Net> model,
    torch::Device device,
    DataLoader& data_loader,
    size_t dataset_size) {
  torch::NoGradGuard no_grad;
  model->eval();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    auto output = model->forward(data);
    // test_loss += torch::nll_loss(
    //                  output,
    //                  targets,
    //                  /*weight=*/{},
    //                  at::Reduction::Sum)
    //                  .template item<float>();
    test_loss += torch::nll_loss(output,targets).template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::printf(
      "\nTest set: Average loss: %.4f | Accuracy: %.3f\n",
      test_loss,
      static_cast<double>(correct) / dataset_size);
}


auto main() -> int {
  torch::manual_seed(1);

  torch::DeviceType device_type;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA available! Training on GPU." << std::endl;
    device_type = torch::kCUDA;
  } else {
    std::cout << "Training on CPU." << std::endl;
    device_type = torch::kCPU;
  }
  torch::Device device(device_type);

  // Net model;
  // model.to(device);
  	
  std::shared_ptr<Net> model = std::make_shared<Net>();// 智能指针，会自动free
  // Net* model =new model(); // 这种需要手动free, delete model;
  model->to(device);

  auto train_dataset = torch::data::datasets::MNIST(kDataRoot)// 0～1
                           .map(torch::data::transforms::Normalize<>(0.1307, 0.3081)) // 均值，方差 or 0.5, 0.5
                           .map(torch::data::transforms::Stack<>());//排序
  const size_t train_dataset_size = train_dataset.size().value();
  // const int64_t batches_per_epoch =
  //   std::ceil(dataset.size().value() / static_cast<double>(kTrainBatchSize));
  // auto train_loader =
  //     torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
  //         std::move(train_dataset), kTrainBatchSize);

  auto train_loader = torch::data::make_data_loader(
    std::move(train_dataset),
    torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(2));

  auto test_dataset = torch::data::datasets::MNIST(
                          kDataRoot, torch::data::datasets::MNIST::Mode::kTest)
                          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                          .map(torch::data::transforms::Stack<>());
  const size_t test_dataset_size = test_dataset.size().value();
  auto test_loader =
      torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);

  // torch::optim::SGD optimizer(
  //     model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.5));
  torch::optim::Adam optimizer(
      model->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

  if (kRestoreFromCheckpoint) {
    torch::load(model, "model-checkpoint.pt");
    // torch::load(optimizer, "model-optimizer-checkpoint.pt");
  }

  for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
    train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
    test(model, device, *test_loader, test_dataset_size);

    // 保存模型参数
    // Checkpoint the model and optimizer state.
    torch::save(model, "model-checkpoint.pt"); // 
    // torch::save(model->parameters(), "model.pt");
    // torch::save(model->named_parameters(),"model.pt");
    // torch::save(optimizer, "model-optimizer-checkpoint.pt");
  }

}
