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

#include "hybrid_embedding_utils.hpp"

#include <algorithm>
#include <iostream>
#include <vector>


namespace {

template <typename dtype>
void download_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, CudaStream_t stream) {
  size_t tensor_size = tensor.get_size_in_bytes() / sizeof(dtype);
  h_tensor.resize(tensor_size);
  CK_CUDA_THROW(cudaStreamSynchronize(stream));
  CK_CUDA_THROW(cudaMemcpy(
    h_tensor.data(), tensor.get_ptr(), 
    tensor.get_size_in_bytes(), cudaMemcpyDeviceToHost, stream)); 
  CK_CUDA_THROW(cudaStreamSynchronize(stream));
}


template <typename dtype>
void upload_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, CudaStream_t stream) {
  CK_CUDA_THROW(cudaStreamSynchronize(stream));
  CK_CUDA_THROW(cudaMemcpyAsync(
    tensor.get_ptr(), h_tensor.data(), 
    h_tensor.size()*sizeof(dtype), cudaMemcpyHostToDevice, stream));
  CK_CUDA_THROW(cudaStreamSynchronize(stream));
}

}

namespace HugeCTR {

/// @brief init_model calculates the optimal number of frequent categories 
///        given the calibration of the all-to-all and all-reduce.
template<dtype>
void HybridEmbeddingModel::init_model(
    const CalibrationData& calibration,
    const HybridEmbeddingData<dtype>& embedding_data,
    Tensor2<dtype> samples
) {

  if (calibration.all_to_all_times.size() > 0) {
    // calibration is given, perform fully optimized hybrid model
    CK_THROW(Error_t::WrongInput, "initialization hybrid model from communication calibration not available yet");
  } else {
      Tensor2<dtype> samples = embedding_data.samples;
      size_t num_nodes = (double) num_gpus_per_node.size();
      size_t num_gpus = (size_t) 0;
      for (size_t i = 0; i < num_gpus_per_node.size(); ++i) {
        num_gpus += num_gpus_per_nodex[i];
      }

      size_t batch_size = embedding_data.batch_size;
      size_t num_gpus = embedding_data.num_networks;
      size_t num_tables = embedding_data.table_sizes.size();
      size_t num_iterations = embedding_data.num_iterations;

      size_t num_samples = batch_size * num_iterations * num_tables;

      // Use threshold to determine number of frequent categories,
      // calculates optimal number of frequent categories when the all-to-all 
      // and all-reduce are both bandwidth limited.
      // 
      // max_._bandwidth is the maximum achieved algorithm bandwidth
      double all_reduce_bandwidth = calibration.max_all_reduce_bandwidth;
      double all_to_all_bandwidth = calibration.max_all_to_all_bandwidth;

      float count_threshold = 1.f;
      double count_threshold = calibration.calculate_threshold(
        communication_type, batch_size, num_networks, num_iterations, num_tables);


      // samples, num_samples, categories_sorted, counts_sorted

      // sort samples by category
      // per category get count
      // sort category by count


      sort_categories_by_count(
          samples, num_samples, categories_sorted, counts_sorted);

      // initialize frequent_category_index
      // find first element smaller than n_threshold
      std::vector<uint32_t> h_counts_sorted;
      download_tensor(h_counts_sorted, counts_sorted, stream);

      for (size_t i = 0; i < h_counts_sorted; ++i) {
        (float) h_counts_sorted[i] / (float) 
      }

      // 
      // initialize category_location
      // node_id, gpu_id, buffer_index
      std::fill(category_location.begin(), category_location.end(), num_categories);
      
  }
}

// template definitions
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils_includes.cuh"
}