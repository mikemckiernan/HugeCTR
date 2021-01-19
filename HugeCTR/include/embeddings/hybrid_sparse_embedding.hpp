/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/tensor2.hpp"

#include <vector>

namespace HugeCTR {

template <typename dtype, typename TypeEmbedding>
class HybridSparseEmbedding : public IEmbedding {
private:

  // the hybrid model
  dtype num_frequent_;
  dtype num_categories_;

  std::vector<uint32_t> category_frequent_index_;
  std::vector<dtype> frequent_categories_;

  // gpu model instances
  std::vector<FrequentEmbedding<dtype, TypeEmbedding>> frequent_embeddings_;
  std::vector<InfrequentEmbedding<dtype, TypeEmbedding>> infrequent_embeddings_;

  // models_ and data_ are replications of the model and input data on each gpu
  // the HybridSparseEmbedding class manages it's scope / frees the memory.
  std::vector<HybridEmbeddingModel<dtype>> model_;
  std::vector<HybridEmbeddingData<dtype>> data_;

public:
  HybridSparseEmbedding();
  ~HybridSparseEmbedding();

  void forward(bool is_train) override;
  void backward() override;
  void update_params() override;
  void init_params() override;
  void load_parameters(std::istream& stream) override;
  void dump_parameters(std::ostream& stream) const override;
  void set_learning_rate(float lr) override;
  size_t get_params_num() const override;
  size_t get_vocabulary_size() const override;
  size_t get_max_vocabulary_size() const override;
  std::vector<TensorBag2> get_train_output_tensors() const override;
  std::vector<TensorBag2> get_evaluate_output_tensors() const override;

  void initialize_model();
};


struct CalibrationInitializationData {
  CalibrationInitializationData() {}
  ~CalibrationInitializationData() {}

  // Calibration all-to-all : 
  //   the following two arrays map data sizes to all-to-all times / latencies.
  std::vector<double> all_to_all_data_size; // data size of message per gpu
  std::vector<double> all_to_all_times;     // calibrated all-to-all times

  // Calibration all-reduce : 
  //   the following two arrays map data sizes to all-to-all times / latencies.
  std::vector<double> all_reduce_data_size; // data size of message per gpu
  std::vector<double> all_reduce_times;     // calibrated all-reduce times

  // Alternative calibration: (if no calibration provided)
  //   the threshold for frequent categories is calculated from maximum bandwidths
  //   for the all-reduce and all-to-all respectively. 
  //   This approximation assumes that the communications are bandwidth limited.
  double max_all_reduce_bandwidth; // algorithm bandwidth all-reduce [data size message per gpu in bytes / sec]
  double max_all_to_all_bandwidth; // algorithm bandwidth all-to-all [data size message per gpu in bytes / sec]
};


template <typename dtype>
struct HybridEmbeddingModel {
public:
  HybridEmbeddingModel() {}
  ~HybridEmbeddingModel() {}

  dtype num_frequent;
  dtype num_categories;

  uint32_t node_id;
  uint32_t gpu_id;

  std::vector<uint32_t> num_gpus_per_node; // number of gpus for each node, .size() == number of nodes

  Tensor2<dtype> category_frequent_index;  // is this category a frequent category? => location in cache
  Tensor2<dtype> category_location;        // is this an infrequent category? 
                                           // => location of where the categories are stored
  void init_model(
    const CalibrationInitializationData& calibration,
    Tensor2<dtype> samples
    );
};


template <typename dtype>
struct HybridEmbeddingData {
  std::vector<uint32_t> table_sizes;
  size_t batch_size;
  size_t num_networks;

  // pointer to raw data: iteration from data reader
  Tensor2<dtype> data;
  // flattened data
  Tensor2<dtype> samples;

  cudaStream_t stream;

  // flatten raw input data
  void flatten_samples(cudaStream_t stream);
};


// One FrequentEmbedding instance per gpu
template <typename dtype, typename TypeEmbedding>
class FrequentEmbedding {
  // copy of the model parameters and the input data
  HybridEmbeddingModel<dtype> model_;
  HybridEmbeddingData<dtype> data_;

  // locally stored embedding vectors for the data-parallel part of the embedding
  Tensor2<TypeEmbedding> frequent_embedding_vectors_;
  // locally stored reduced gradients: input for the all-reduce
  Tensor2<TypeEmbedding> frequent_partial_gradients_;

  cudaStream_t stream_;

  void init();
public:
  FrequentEmbedding() {}
  ~FrequentEmbedding() {}

  void initialize_embedding_vectors();

  void all_reduce();
  void update();
};


template <typename dtype, typename TypeEmbedding>
class InfrequentEmbedding {
  // copy of the model parameters and the input data, managed by HybridSparseEmbedding
  HybridEmbeddingModel<dtype> model_;
  HybridEmbeddingData<dtype> data_;

  Tensor2<TypeEmbedding> infrequent_embedding_vectors_;

  // forward-send, backward-receive
  Tensor2<dtype> model_indices_;
  // forward-receive, backward-send
  Tensor2<dtype> network_indices_;

  cudaStream_t stream_;

public:
  InfrequentEmbedding();
  ~InfrequentEmbedding();  

  void initialize_embedding_vectors();

  void all_to_all_forward();
  void all_to_all_backward();
};


}  // namespace HugeCTR