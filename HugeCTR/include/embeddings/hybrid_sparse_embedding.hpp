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
  std::vector<InfrequentEmbedding<dtype, TypeEmbedding>> infrequent_embedding_;

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
};


template <typename dtype>
struct HybridEmbeddingModel {
public:
  HybridEmbeddingModel() {}
  ~HybridEmbeddingModel() {}

  dtype num_frequent;
  dtype num_categories;

  Tensor2<dtype> category_frequent_index;
  Tensor2<uint32_t> category_location;
};

template <typename dtype>
struct HybridEmbeddingData {
  dtype num_tables;
  std::vector<uint32_t> table_sizes;

  // pointer to raw data: iteration from data reader
  Tensor2<dtype> data;
  // flattened data
  Tensor2<dtype> samples;

  // flatten raw input data
  void calculate_samples();
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

  void init();
public:
  FrequentEmbedding() {}
  ~FrequentEmbedding() {}

  void all_reduce();
  void update();
};


template <typename dtype, typename TypeEmbedding>
class InfrequentEmbedding {
  Tensor2<TypeEmbedding> infrequent_embedding_vectors_;

  // forward-send, backward-receive
  Tensor2<dtype> model_indices_;
  // forward-receive, backward-send
  Tensor2<dtype> network_indices_;

public:
  InfrequentEmbedding();
  ~InfrequentEmbedding();  

  void all_to_all_forward();
  void all_to_all_backward();
};


}  // namespace HugeCTR