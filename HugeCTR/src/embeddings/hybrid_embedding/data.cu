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

#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
 
#include <algorithm>
#include <iostream>
#include <vector>

namespace hybrid_embedding {


/// data_to_unique_categories converts the argument 'data' and stores
///        the result in member variable 'samples'.
///        Per network, the columns corresponding to embedding tables 
///        are concatenated and categories get an unique index / label.
template <typename dtype>
void Data::data_to_unique_categories(
    Tensor2<dtype> data,
    cudaStream_t stream
) {
  /// === TODO: PERFORM ON GPU ===
  /// ============================
  // std::cout << "WARNING: data_to_unique_categories() needs to be placed on the GPU!" << std::endl;
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

  upload_tensor(h_samples, samples, stream);
  /// ============================
  /// ============================
}


template class Data<uint32_t>;
template class Data<size_t>;
}