/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <optimizer.hpp>
#include <tensor2.hpp>
#include <vector>

namespace HugeCTR {

class IEmbedding {
 public:
  virtual ~IEmbedding() = default;
  /**
   * The forward propagation of embedding layer.
   */
  virtual void forward(bool is_train) = 0;
  /**
   * The first stage of backward propagation of embedding layer,
   * which only computes the wgrad by the dgrad from the top layer.
   */
  virtual void backward() = 0;
  /**
   * The second stage of backward propagation of embedding layer, which
   * updates the embedding table weights by wgrad(from backward()) and
   * optimizer.
   */
  virtual void update_params() = 0;
  /**
   * Initialize the embedding table
   */
  virtual void init_params() = 0;
  /**
   * Read the embedding table from the stream on the host, and
   * upload it onto multi-GPUs global memory.
   * @param stream the host file stream for reading data from.
   */
  virtual void load_parameters(std::istream& stream) = 0;
  /**
   * Download the embedding table from multi-GPUs global memroy to CPU memory
   * and write it to the stream on the host.
   * @param stream the host file stream for writing data to.
   */
  virtual void dump_parameters(std::ostream& stream) const = 0;
  virtual void set_learning_rate(float lr) = 0;
  /**
   * Get the total size of embedding tables on all GPUs.
   */
  virtual size_t get_params_num() const = 0;
  virtual size_t get_vocabulary_size() const = 0;
  virtual size_t get_max_vocabulary_size() const = 0;
  virtual std::vector<TensorBag2> get_train_output_tensors() const = 0;
  virtual std::vector<TensorBag2> get_evaluate_output_tensors() const = 0;
};

class IHashEmbedding : public IEmbedding {
 public:
  virtual void check_overflow() const = 0;
};

template <typename TypeKey, typename TypeEmbedding>
class IEmbeddingForUnitTest {
  /**
   * Get the forward() results from GPUs and copy them to the host pointer
   * embedding_feature. This function is only used for unit test.
   * @param embedding_feature the host pointer for storing the forward()
   * results.
   */
  virtual void get_forward_results(bool is_train, Tensor2<TypeEmbedding>& embedding_feature) = 0;
  /**
   * Get the backward() results from GPUs and copy them to the host pointer
   * wgrad. The wgrad on each GPU should be the same. This function is only
   * used for unit test.
   * @param wgrad the host pointer for stroing the backward() results.
   * @param devIndex the GPU device id.
   */
  virtual void get_backward_results(Tensor2<TypeEmbedding>& wgrad, int devIndex) = 0;
  /**
   * Get the update_params() results(the hash table, including hash_table_keys
   * and hash_table_values) from GPUs and copy them to the host pointers.
   * This function is only used for unit test.
   * @param hash_table_key the host pointer for stroing the hash table keys.
   * @param hash_table_value the host pointer for stroing the hash table values.
   */
  virtual void get_update_params_results(Tensor2<TypeKey>& hash_table_key,
                                         Tensor2<float>& hash_table_value) = 0;
};

class IEmbeddingForTensorFlowPlugin {
 public:
  virtual ~IEmbeddingForTensorFlowPlugin() = default;
  virtual void get_forward_results_tf(const bool is_train, const bool on_gpu,
                                      void* const forward_result) = 0;
  virtual cudaError_t update_top_gradients(const bool on_gpu, const void* const top_gradients) = 0;
};

template <typename TypeEmbedding>
struct SparseEmbeddingHashParams {
  size_t train_batch_size;  // batch size
  size_t evaluate_batch_size;
  size_t max_vocabulary_size_per_gpu;   // max row number of hash table for each gpu
  std::vector<size_t> slot_size_array;  // max row number for each slot
  size_t embedding_vec_size;            // col number of hash table value
  size_t max_feature_num;               // max feature number of all input samples of all slots
  size_t slot_num;                      // slot number
  int combiner;                         // 0-sum, 1-mean
  OptParams<TypeEmbedding> opt_params;  // optimizer params
};

struct HybridSparseEmbeddingCategoryItem {
  size_t index;
  size_t global_gpu_id;
};

template <typename TypeEmbedding>
struct HybridSparseEmbeddingParams {
  size_t train_batch_size;
  size_t evaluate_batch_size;
  std::vector<HybridSparseEmbeddingCategoryItem> categories;
  size_t embedding_vec_size;
  size_t slot_num;                      // slot number
  OptParams<TypeEmbedding> opt_params;  // optimizer params
};

template <typename TypeEmbedding>
class IEmbeddingForPrefetcher {
 public:
  virtual void load_parameters(const TensorBag2& keys, const Tensor2<float>& embeddings,
                               size_t num) = 0;
  virtual void dump_parameters(TensorBag2 keys, Tensor2<float>& embeddings, size_t* num) const = 0;
  virtual void reset() = 0;
  virtual const SparseEmbeddingHashParams<TypeEmbedding>& get_embedding_params() const = 0;
};

}  // namespace HugeCTR
