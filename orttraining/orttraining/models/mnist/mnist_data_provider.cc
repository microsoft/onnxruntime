// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mnist_reader/mnist_reader.hpp"
#include "mnist_reader/mnist_utils.hpp"
#include "mnist_data_provider.h"
#include "orttraining/models/runner/training_util.h"

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

typedef uint8_t Label;
typedef vector<uint8_t> Image;

pair<vector<vector<float>>, vector<vector<float>>> NormalizeData(const vector<Image>& images, const vector<Label>& labels) {
  vector<vector<float>> normalized_images;
  for (size_t i = 0; i < images.size(); i++) {
    // Binarize the image.
    vector<float> normalized_image(images[i].begin(), images[i].end());
    for (size_t j = 0; j < images[i].size(); j++) {
      if (images[i][j] > 0) {
        normalized_image[j] = 1.0f;
      }
    }
    normalized_images.emplace_back(move(normalized_image));
  }

  vector<vector<float>> one_hot_labels;
  for (size_t i = 0; i < labels.size(); i++) {
    vector<float> one_hot_label(10, 0.0f);
    one_hot_label[labels[i]] = 1.0f;  //one hot
    one_hot_labels.emplace_back(move(one_hot_label));
  }
  return make_pair(normalized_images, one_hot_labels);
}

void ConvertData(const vector<vector<float>>& images,
                 const vector<vector<float>>& labels,
                 const vector<int64_t>& image_dims,
                 const vector<int64_t>& label_dims,
                 DataSet& data_set,
                 size_t shard_index = 0,
                 size_t total_shard = 1) {
  for (size_t i = 0; i < images.size(); ++i) {
    if (i % total_shard == shard_index) {
      MLValue imageMLValue;
      TrainingUtil::CreateCpuMLValue(image_dims, images[i], &imageMLValue);
      MLValue labelMLValue;
      TrainingUtil::CreateCpuMLValue(label_dims, labels[i], &labelMLValue);

      data_set.AddData(make_unique<vector<MLValue>>(vector<MLValue>{imageMLValue, labelMLValue}));
    }
  }
}

void PrepareMNISTData(const string& data_folder,
                      const vector<int64_t>& image_dims,
                      const vector<int64_t>& label_dims,
                      DataSet& training_data,
                      DataSet& test_data,
                      size_t shard_index,
                      size_t total_shard) {
  ORT_ENFORCE(shard_index < total_shard, "shard_index must be 0~", total_shard - 1);

  printf("Loading MNIST data from folder %s\n", data_folder.c_str());
  mnist::MNIST_dataset<std::vector, Image, Label> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(data_folder);

  printf("Preparing data ...\n");
  auto training_images_labels = NormalizeData(dataset.training_images, dataset.training_labels);
  auto test_images_labels = NormalizeData(dataset.test_images, dataset.test_labels);
  ConvertData(training_images_labels.first, training_images_labels.second, image_dims, label_dims, training_data,
              shard_index, total_shard);
  ConvertData(test_images_labels.first, test_images_labels.second, image_dims, label_dims, test_data);  // use full test set
  printf("Preparing data: done\n");
  printf("#training set size = %d \n", static_cast<int>(training_data.NumSamples()));
  printf("#test set size = %d \n", static_cast<int>(test_data.NumSamples()));
}
