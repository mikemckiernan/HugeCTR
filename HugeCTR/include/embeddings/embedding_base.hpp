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
#include <embedding.hpp>
#include <general_buffer2.hpp>
#include <resource_manager.hpp>

namespace HugeCTR {

/**
 * @brief The base class of embedding layers.
 *
 * This class has responsibility to get the tensors
 * from the former layer and to allocate buffers for the output tensors of the embedding
 * layer. As a base class, it declares all the virtual functions that need to be implemented
 * in embedding layers' training process, including forward propagation and backward
 * propagation. The forward propagation is corresponding to the API forward(). The backward
 * propagation is divided into 2-stage APIs: backward() and update_params(). This class also
 * provides the operations for uploading/downloading embedding models (which is also known as
 * embedding tables) to/from GPUs from/to host file stream, which are named as
 * load_parameters() and dump_parameters().
 */
template <typename T, typename TypeKey, typename TypeEmbedding>
class EmbeddingBase : public T {
  SparseEmbeddingHashParams<TypeEmbedding> embedding_params_; /**< Sparse embedding hash params. */
  std::vector<OptParams<TypeEmbedding>> opt_params_;          /**< Optimizer params. */

  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>>
      bufs_;                                        /**< The buffer for storing output tensors. */
  Tensors2<TypeEmbedding> train_output_tensors_;    /**< The output tensors. */
  Tensors2<TypeEmbedding> evaluate_output_tensors_; /**< The output tensors. */
  Tensors2<TypeKey> train_row_offsets_tensors_; /**< The row_offsets tensors of the input data. */
  Tensors2<TypeKey> train_value_tensors_;       /**< The value tensors of the input data. */
  std::vector<std::shared_ptr<size_t>> train_nnz_array_;
  Tensors2<TypeKey>
      evaluate_row_offsets_tensors_;         /**< The row_offsets tensors of the input data. */
  Tensors2<TypeKey> evaluate_value_tensors_; /**< The value tensors of the input data. */
  std::vector<std::shared_ptr<size_t>> evaluate_nnz_array_;

  std::shared_ptr<ResourceManager> resource_manager_; /**< The GPU device resources. */

 protected:
  size_t get_batch_size(bool is_train) const {
    if (is_train) {
      return embedding_params_.train_batch_size;
    } else {
      return embedding_params_.evaluate_batch_size;
    }
  }

  size_t get_universal_batch_size() const {
    return std::max(embedding_params_.train_batch_size, embedding_params_.evaluate_batch_size);
  }

  size_t get_batch_size_per_gpu(bool is_train) const {
    return get_batch_size(is_train) / resource_manager_->get_global_gpu_count();
  }

  size_t get_batch_size_per_lane(bool is_train) const {
    return get_batch_size(is_train) / resource_manager_->get_local_gpu_count();
  }

  size_t get_universal_batch_size_per_gpu() const {
    return get_universal_batch_size() / resource_manager_->get_global_gpu_count();
  }

  ResourceManager& get_resource_manager() const { return *resource_manager_; }

  const GPUResource& get_local_gpu(int i) const { return *resource_manager_->get_local_gpu(i); }

  const Optimizer_t& get_optimizer() const { return embedding_params_.opt_params.optimizer; }

  const Update_t& get_update_type() const { return embedding_params_.opt_params.update_type; }

  OptParams<TypeEmbedding>& get_opt_params(int i) { return opt_params_[i]; }

  const OptParams<TypeEmbedding>& get_opt_params() const { return embedding_params_.opt_params; }

  size_t get_embedding_vec_size() const { return embedding_params_.embedding_vec_size; }

  size_t get_max_feature_num() const { return embedding_params_.max_feature_num; }

  size_t get_slot_num() const { return embedding_params_.slot_num; }

  int get_combiner() const { return embedding_params_.combiner; }

  size_t get_max_vocabulary_size_per_gpu() const {
    return embedding_params_.max_vocabulary_size_per_gpu;
  }

  const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& get_buffer(size_t i) const {
    return bufs_[i];
  }

  Tensors2<TypeEmbedding>& get_output_tensors(bool is_train) {
    if (is_train) {
      return train_output_tensors_;
    } else {
      return evaluate_output_tensors_;
    }
  }

  const Tensors2<TypeKey>& get_row_offsets_tensors(bool is_train) const {
    if (is_train) {
      return train_row_offsets_tensors_;
    } else {
      return evaluate_row_offsets_tensors_;
    }
  }

  const Tensors2<TypeKey>& get_value_tensors(bool is_train) const {
    if (is_train) {
      return train_value_tensors_;
    } else {
      return evaluate_value_tensors_;
    }
  }

  const std::vector<std::shared_ptr<size_t>>& get_nnz_array(bool is_train) const {
    if (is_train) {
      return train_nnz_array_;
    } else {
      return evaluate_nnz_array_;
    }
  }

 public:
  /**
   * The constructor of Embedding class.
   * @param row_offsets_tensors the row_offsets tensors of the input data(refer to row offset vector
   * in sparse matrix CSR format).
   * @param value_tensors the value tensors of the input data(refer to value vector in sparse matrix
   * CSR format).
   * @param batchsize the batch size of the input data
   * @param slot_num the number of slots of the hash table
   * @param embedding_vec_size the dim size of the embedding feature vector.
   * @param resource_manager the GPU device resource group
   * @param scaler scaler factor for mixed precision
   */
  EmbeddingBase(const Tensors2<TypeKey>& train_row_offsets_tensors,
                const Tensors2<TypeKey>& train_value_tensors,
                const std::vector<std::shared_ptr<size_t>>& train_nnz_array,
                const Tensors2<TypeKey>& evaluate_row_offsets_tensors,
                const Tensors2<TypeKey>& evaluate_value_tensors,
                const std::vector<std::shared_ptr<size_t>>& evaluate_nnz_array,
                const SparseEmbeddingHashParams<TypeEmbedding>& embedding_params,
                const std::shared_ptr<ResourceManager>& resource_manager)
      : embedding_params_(embedding_params),
        train_row_offsets_tensors_(train_row_offsets_tensors),
        train_value_tensors_(train_value_tensors),
        train_nnz_array_(train_nnz_array),
        evaluate_row_offsets_tensors_(evaluate_row_offsets_tensors),
        evaluate_value_tensors_(evaluate_value_tensors),
        evaluate_nnz_array_(evaluate_nnz_array),
        resource_manager_(resource_manager) {
    try {
      // Error check
      if (embedding_params.train_batch_size < 1 || embedding_params.evaluate_batch_size < 1 ||
          embedding_params.slot_num < 1 || embedding_params.embedding_vec_size < 1) {
        CK_THROW_(Error_t::WrongInput, "batchsize < 1 || slot_num < 1 || embedding_vec_size < 1");
      }

      if (embedding_params.embedding_vec_size > 1024) {
        CK_THROW_(Error_t::WrongInput,
                  "the embedding_vec_size can not be more than 1024 in embedding layer");
      }

      size_t total_gpu_count = resource_manager_->get_global_gpu_count();
      size_t local_gpu_count = resource_manager_->get_local_gpu_count();

      if (train_row_offsets_tensors.size() != local_gpu_count ||
          train_value_tensors.size() != local_gpu_count ||
          evaluate_row_offsets_tensors.size() != local_gpu_count ||
          evaluate_value_tensors.size() != local_gpu_count) {
        CK_THROW_(
            Error_t::WrongInput,
            "either row_offsets_tensors.size() or value_tensors.size() isn't local_gpu_count_");
      }

      for (size_t id = 0; id < local_gpu_count; id++) {
        OptParams<TypeEmbedding> opt_params;
        opt_params.optimizer = embedding_params_.opt_params.optimizer;
        opt_params.lr = embedding_params_.opt_params.lr;
        opt_params.update_type = embedding_params_.opt_params.update_type;
        opt_params.scaler = embedding_params_.opt_params.scaler;
        opt_params_.push_back(opt_params);
      }

      assert(bufs_.empty());
      for (size_t i = 0; i < local_gpu_count; i++) {
        std::shared_ptr<GeneralBuffer2<CudaAllocator>> buf =
            GeneralBuffer2<CudaAllocator>::create();
        bufs_.push_back(buf);

        Tensor2<TypeEmbedding> tensor;
        buf->reserve({get_batch_size_per_gpu(true), get_slot_num(), get_embedding_vec_size()},
                     &tensor);
        train_output_tensors_.push_back(tensor);
        buf->reserve({get_batch_size_per_gpu(false), get_slot_num(), get_embedding_vec_size()},
                     &tensor);
        evaluate_output_tensors_.push_back(tensor);
      }

    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
    return;
  }

  /**
   * The declaration for indicating that there is no default copy construtor in this class.
   */
  EmbeddingBase(const EmbeddingBase&) = delete;
  EmbeddingBase& operator=(const EmbeddingBase&) = delete;

  virtual ~EmbeddingBase() = default;

  /**
   * Return the output tensors.
   */
  std::vector<TensorBag2> get_train_output_tensors() const {
    std::vector<TensorBag2> bags;
    for (const auto& t : train_output_tensors_) {
      bags.push_back(t.shrink());
    }
    return bags;
  }

  std::vector<TensorBag2> get_evaluate_output_tensors() const {
    std::vector<TensorBag2> bags;
    for (const auto& t : evaluate_output_tensors_) {
      bags.push_back(t.shrink());
    }
    return bags;
  }

  void set_learning_rate(float lr) {
    for (size_t id = 0; id < resource_manager_->get_local_gpu_count(); id++) {
      opt_params_[id].lr = lr;
    }
  }

  const SparseEmbeddingHashParams<TypeEmbedding>& get_embedding_params() const {
    return embedding_params_;
  }
};
}  // namespace HugeCTR
