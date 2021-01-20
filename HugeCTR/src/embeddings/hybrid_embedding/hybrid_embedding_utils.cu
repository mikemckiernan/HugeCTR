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

#include <vector>
#include <iostream>


template <typename dtype>
void download_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, CudaStream_t stream) {
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


/// @brief flatten_samples converts the member variable 'data' and store 
///        the result in member variable 'samples'. 
///        Per network, the columns corresponding to embedding tables 
///        are concatenated and categories get an unique index / label.
template <typename dtype>
void HybridEmbeddingData::flatten_samples(

    cudaStream_t stream
) {

  std::cout << "WARNING: flatten_samples needs to be placed on the GPU!" << std::endl;
  // TODO : perform conversion by kernel (before start of iteration ? => see below)
  //        for batch_size = 55*1024
  //        batch_size * 26 * 4 / 1600e9 = 3.67 microseconds, 
  // 
  // Remark:
  //        Doesn't need to be before start of kernel. 
  //        Would be nice to have just before calculating indices, since
  //        those would be in L2 cache already.
  std::vector<dtype> h_data;
  download_tensor<dtype>(h_data, data, stream);

  const size_t num_tables = table_sizes.size();
  std::vector<dtype> embedding_offsets(num_tables);
  dtype embedding_offset = (dtype) 0;
  for (size_t embedding = 0; embedding < num_tables; ++embedding) {
    embedding_offsets[embedding] = embedding_offset;
    embedding_offset += table_sizes[embedding];
  }

  // keep order of input samples, convert each data field such that categories
  // from different categorical features have different label / index
  size_t indx = 0;
  std::vector<dtype> h_samples(num_tables * batch_size);
  for (size_t i = 0; i < network_batch_size; ++i) {
    for (size_t embedding=0; embedding < num_tables; ++embedding) {
      h_samples[indx] = h_data[indx] + embedding_offsets[embedding];
      indx++;
    }
  }

  // TODO : remove
  upload_tensor(h_samples, samples, stream);
}


/// @brief init_model calculates the optimal number of frequent categories 
///        given the calibration of the all-to-all and all-reduce.
template<dtype>
void HybridEmbeddingModel::init_model(
    const CalibrationData& calibration,
    const HybridEmbeddingData<dtype>& embedding_data
) {

  if (calibration.all_to_all_times.size() > 0) {
    // calibration is given, perform fully optimized hybrid model
    CK_THROW(Error_t::WrongInput, "initialization hybrid model from communication calibration not available yet");
  } else {
      Tensor2<dtype> samples = embedding_data.samples;
      size_t num_nodes = (double) num_gpus_per_node.size();

      // Use threshold to determine number of frequent categories,
      // calculates optimal number of frequent categories when the all-to-all 
      // and all-reduce are both bandwidth limited.
      double all_reduce_bandwidth = calibration.max_all_reduce_bandwidth;
      double all_to_all_bandwidth = calibration.max_all_to_all_bandwidth;
      n_threshold = all_to_all_bandwidth / all_reduce_bandwidth 
                    * (double) num_nodes / ((double) num_nodes - 1.);

      // samples, num_samples, categories_sorted, counts_sorted

      // sort samples by category
      // per category get count
      // sort category by count

      sort_categories_by_count(
          samples, num_samples, categories_sorted, counts_sorted);

      

  }
}

// template definitions
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils_include.cuh"
