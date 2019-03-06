// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include "core/graph/model.h"
#include "core/common/logging/logging.h"
#include "core/framework/environment.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/training/training_session.h"
#include "core/training/training_optimizer.h"
#include "mnist_reader/mnist_reader.hpp"
#include "mnist_reader/mnist_utils.hpp"

#include <random>

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

const static float LEARNING_RATE = 0.5f;
const static int MAX_STEPS = 2000;
const static int BATCH_SIZE = 100;
const static int NUM_CLASS = 10;

const static char* ORIGINAL_MODEL_PATH = "mnist_fc_model.onnx";
const static char* GENERATED_MODEL_WITH_COST_PATH = "mnist_fc_model_with_cost.onnx";
const static char* BACKWARD_MODEL_PATH = "mnist_fc_model_bw.onnx";
const static char* TRAINED_MODEL_PATH = "mnist_fc_model_trained.onnx";
const static char* TRAINED_MODEL_WITH_COST_PATH = "mnist_fc_model_with_cost_trained.onnx";
const static char* MNIST_DATA_PATH = "mnist_data";

#define TERMINATE_IF_FAILED(action)                                    \
  {                                                                    \
    auto status = action;                                              \
    if (!status.IsOK()) {                                              \
      LOGF_DEFAULT(ERROR, "Failed:%s", status.ErrorMessage().c_str()); \
      return -1;                                                       \
    }                                                                  \
  }

typedef uint8_t Label;
typedef vector<uint8_t> Image;

class DataSet {
 public:
  DataSet(vector<Image> images, vector<Label> labels) : num_samples_(int(images.size())),
                                                        images_(images),
                                                        labels_(labels),
                                                        epochs_completed_(0),
                                                        index_in_epoch_(0) {
    for (int i = 0; i < images.size(); i++) {
      // Binarize the image.
      vector<float> normalized_image(images[i].begin(), images[i].end());
      for (int j = 0; j < images[i].size(); j++) {
        if (images[i][j] > 0) {
          normalized_image[j] = 1.0f;
        }
      }
      normalized_images_.push_back(normalized_image);
    }

    for (int i = 0; i < labels.size(); i++) {
      vector<float> one_hot_label(10, 0.0f);
      one_hot_label[labels[i]] = 1.0f;  //one hot
      one_hot_labels_.push_back(one_hot_label);
    }
  }

  pair<vector<vector<float>>, vector<vector<float>>> NextBatch(int batch_size, bool shuffle = true) {
    // shuffle for the first epoch
    if (epochs_completed_ == 0 && index_in_epoch_ == 0 && shuffle) {
      Shuffle();
    }

    pair<vector<vector<float>>, vector<vector<float>>> batch;
    // go to next epoch
    int start = index_in_epoch_;
    if (start + batch_size > num_samples_) {
      epochs_completed_++;

      int rest_num_samples = num_samples_ - start;
      for (int i = start; i < num_samples_; i++) {
        batch.first.push_back(normalized_images_[i]);
        batch.second.push_back(one_hot_labels_[i]);
      }

      if (shuffle) {
        Shuffle();
      }

      start = 0;
      int end = batch_size - rest_num_samples;
      for (int i = start; i < end; i++) {
        batch.first.push_back(normalized_images_[i]);
        batch.second.push_back(one_hot_labels_[i]);
      }
      index_in_epoch_ = end;
    } else {
      for (int i = index_in_epoch_; i < index_in_epoch_ + batch_size; i++) {
        batch.first.push_back(normalized_images_[i]);
        batch.second.push_back(one_hot_labels_[i]);
      }

      index_in_epoch_ += batch_size;
    }

    return batch;
  }

  int NumSamples() {
    return num_samples_;
  }

 private:
  void Shuffle() {
    auto rng1 = std::default_random_engine{};
    auto rng2 = rng1;
    std::shuffle(normalized_images_.begin(), normalized_images_.end(), rng1);
    std::shuffle(one_hot_labels_.begin(), one_hot_labels_.end(), rng2);
  }

  int num_samples_;
  vector<Image> images_;
  vector<Label> labels_;

  vector<vector<float>> normalized_images_;
  vector<vector<float>> one_hot_labels_;

  int epochs_completed_;
  int index_in_epoch_;
};

template <typename T>
static void CreateMLValue(AllocatorPtr alloc,
                          const std::vector<int64_t>& dims,
                          const std::vector<T>& value,
                          MLValue* p_mlvalue) {
  TensorShape shape(dims);
  auto location = alloc->Info();
  auto element_type = DataTypeImpl::GetType<T>();
  void* buffer = alloc->Alloc(element_type->Size() * shape.Size());
  if (value.size() > 0) {
    memcpy(buffer, &value[0], element_type->Size() * shape.Size());
  }

  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              location,
                                                              alloc);
  p_mlvalue->Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

AllocatorPtr GetAllocator() {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  return cpu_provider.GetAllocator(0, OrtMemTypeDefault);
}

vector<NameMLValMap> fill_feed_dict(const pair<vector<vector<float>>, vector<vector<float>>>& batch) {
  vector<NameMLValMap> feeds;

  const vector<vector<float>>& images = batch.first;
  const vector<vector<float>>& labels = batch.second;

  vector<int64_t> image_dims = {1, 784};
  vector<int64_t> label_dims = {1, 10};

  for (int i = 0; i < images.size(); i++) {
    MLValue imageMLValue;
    CreateMLValue(GetAllocator(), image_dims, images[i], &imageMLValue);
    MLValue labelMLValue;
    CreateMLValue(GetAllocator(), label_dims, labels[i], &labelMLValue);

    feeds.push_back(NameMLValMap({{"X", imageMLValue}, {"labels", labelMLValue}}));
  }

  return feeds;
}

template <typename SessionType>
void Evaluate(SessionType& sess, DataSet& data_set, bool use_full_set = false) {
  int true_count = 0;
  int num_examples = use_full_set ? data_set.NumSamples() : 1000;

  vector<NameMLValMap> batch = fill_feed_dict(data_set.NextBatch(num_examples));

  vector<string> output_names = {"predictions", "loss"};
  vector<MLValue> fetches;

  for (int i = 0; i < batch.size(); i++) {
    Status s = sess.Run(batch[i], output_names, &fetches);

    const float* prediction_data = fetches[0].Get<Tensor>().template Data<float>();

    auto max_class_index = std::distance(prediction_data,
                                         std::max_element(prediction_data, prediction_data + NUM_CLASS));

    const float* label_data = batch[i]["labels"].Get<Tensor>().template Data<float>();

    // todo:  better way to convert to int
    if (int(label_data[max_class_index]) == 1) {
      true_count++;
    }
  }

  float precision = float(true_count) / num_examples;

  printf("Num examples: %d Num correct: %d  Precision: %0.04f \n", num_examples, true_count, precision);
}

float GetLossValue(const vector<string>& fw_output_names, const vector<MLValue>& fw_fetches, const std::string& loss_name);

int main(int /*argc*/, char* /*args*/[]) {
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING, false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  unique_ptr<Environment> env;
  TERMINATE_IF_FAILED(Environment::Create(env));
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = {};

  // Step 0: Read MNIST data
  bool load_mnist_data = true;
  if (load_mnist_data) {
    dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_PATH);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;
  }
  DataSet training_set(dataset.training_images, dataset.training_labels);
  DataSet testing_set(dataset.test_images, dataset.test_labels);

  // Step 1: Load the model and generate gradient graph in a training session.
  SessionOptions so;
  TrainingSession training_session{so};
  TERMINATE_IF_FAILED(training_session.Load(ORIGINAL_MODEL_PATH));
  TERMINATE_IF_FAILED(training_session.AddLossFuncion({"MeanSquaredError", "predictions", "labels", "loss"}));
  TERMINATE_IF_FAILED(training_session.Save(GENERATED_MODEL_WITH_COST_PATH,
                                            TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));
  TERMINATE_IF_FAILED(training_session.BuildGradientGraph({"W1", "W2", "W3", "B1", "B2", "B3"}, "loss"));
  TERMINATE_IF_FAILED(training_session.Save(BACKWARD_MODEL_PATH,
                                            TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS));
  TERMINATE_IF_FAILED(training_session.Initialize());

  Optimizer<GradientDescent> optimizer(training_session,
                                       {LEARNING_RATE, GetAllocator()});

  cout << "Before training" << endl;
  Evaluate(training_session, testing_set);

  // prepare output names
  auto output_names_include_gradients = training_session.GetModelOutputNames();
  vector<string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  // loop through the data
  for (size_t batch_index = 0; batch_index < MAX_STEPS; ++batch_index) {
    NameMLValMap old_weight;
    vector<NameMLValMap> grads_batch;
    float total_loss = 0;

    // train for a mini batch
    vector<NameMLValMap> training_batch = fill_feed_dict(training_set.NextBatch(BATCH_SIZE));
    for (const auto& fw_feeds : training_batch) {
      vector<MLValue> gradient_fetches;  // All gradients and loss are here, 1:1 mapping to the above training_output_names.

      TERMINATE_IF_FAILED(training_session.Run(fw_feeds, training_output_names, &gradient_fetches));

      // Gradient descent: update weights in the modified model, with the output of Step 5
      // Here we modify the in-memory MLValue so that next training iteration can be run without model saving and reloading.
      // TODO: modify the graph_proto so that the new weights could be saved.

      // Accumulated grads from multi run.
      NameMLValMap grad;
      for (int i = 0; i < training_output_names.size(); i++) {
        if (training_output_names[i] == "loss") continue;
        if (training_output_names[i] == "predictions") continue;

        grad.insert(make_pair(training_output_names[i], gradient_fetches[i]));
      }
      grads_batch.emplace_back(grad);

      total_loss += GetLossValue(training_output_names, gradient_fetches, "loss");
    }  // end of one mini-batch

    // Print some info when reaching the end of the batch.
    cout << "batch / steps: " << batch_index << "/" << MAX_STEPS << "\n"
         << "avg loss: " << to_string(total_loss / BATCH_SIZE) << "\n"
         << endl;

    // After this call, training_session will have the updated weights ready for next Run().
    optimizer.Optimize(grads_batch);

    Evaluate(training_session, testing_set);

    // Save the model at the end of training, with the latest weights
    if (batch_index == MAX_STEPS - 1) {
      TERMINATE_IF_FAILED(training_session.Save(TRAINED_MODEL_WITH_COST_PATH,
                                                TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));
      TERMINATE_IF_FAILED(training_session.Save(TRAINED_MODEL_PATH,
                                                TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS));
    }
  }

  //Load and test the trained model.
  SessionOptions test_so;
  InferenceSession test_session{test_so};

  TERMINATE_IF_FAILED(test_session.Load(TRAINED_MODEL_PATH));
  TERMINATE_IF_FAILED(test_session.Initialize());
  cout << "\nNow testing the saved model" << endl;
  Evaluate(test_session, testing_set, true /*use full test set*/);

  return 0;
}

float GetLossValue(const vector<string>& fw_output_names, const vector<MLValue>& fw_fetches, const std::string& loss_name) {
  float loss = 0.0f;
  size_t pos = std::distance(fw_output_names.begin(), std::find(fw_output_names.begin(), fw_output_names.end(), loss_name));
  if (pos < fw_output_names.size()) {
    loss = *fw_fetches[pos].Get<Tensor>().Data<float>();
  }
  return loss;
}
