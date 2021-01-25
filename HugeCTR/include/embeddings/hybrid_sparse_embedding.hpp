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

#include <vector>

#include "HugeCTR/include/embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/calibration_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/communication.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/frequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/infrequent_embedding.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/statistics.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

using namespace HugeCTR::hybrid_embedding;

namespace HugeCTR {

///
/// Interface class for the hybrid embedding to HugeCTR. It is responsible for
/// persistent gpu memory allocation.
///
template <typename dtype, typename emtype>
class HybridSparseEmbedding : public IEmbedding {
 private:
  // Embedding models, one instance per frequent and the infrequent embedding
  // for each mlp-network in the train session.
  //
  // data-parallel embedding model
  std::vector<FrequentEmbedding<dtype, emtype>> frequent_embeddings_;
  // model-parallel embedding model
  std::vector<InfrequentEmbedding<dtype, emtype>> infrequent_embeddings_;
  // performs the communication scheme
  std::vector<Communication<dtype, emtype>> communication_;

  // model_, data_, calibration_ and statistics_ are replications of the model
  // and input data on each gpu. The HybridSparseEmbedding class manages
  // it's scope / frees the memory.
  std::vector<Model<dtype>> model_;
  std::vector<Data<dtype>> data_;
  std::vector<CalibrationData> calibration_;
  std::vector<Statistics<dtype>> statistics_;

 public:
  HybridSparseEmbedding() {}
  ~HybridSparseEmbedding() {}

  void initialize_model();

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

}  // namespace HugeCTR