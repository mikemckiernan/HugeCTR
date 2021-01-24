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

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
 
namespace HugeCTR {


/// init_model calculates the optimal number of frequent categories 
/// given the calibration of the all-to-all and all-reduce.
template<dtype>
void HybridEmbeddingModel::init_model(
  CommunicationType communication_type,
  CalibrationData<dtype> calibration,
  HybridEmbeddingStatistics<dtype> statistics,
  HybridEmbeddingData<dtype> data,
  cudaStream_t stream
) {

  // calculate the total number of categories
  num_categories = (dtype) 0;
  for (size_t i = 0; i < data.table_sizes.size(); ++i) {
    num_categories += data.table_sizes[i];
  }

  // determine the number of frequent categories

  // create the top categories sorted by count
  Tensor2<dtype> samples = data.samples;
  EmbeddingStatistics statistics(samples.get_size_in_bytes() / sizeof(dtype));
  statistics.calculate_statistics(samples, stream);
  // from the sorted count, determine the number of frequent categories
  num_frequent = calibration.calculate_num_frequent_categories(
      communication_type, calibration, statistics, data, stream);

  /// === TODO: PERFORM ON GPU ===
  /// ============================
  std::vector<dtype> h_categories_sorted;
  std::vector<uint32_t> h_counts_sorted;
  download_tensor(h_categories_sorted, statistics.categories_sorted, stream);
  download_tensor(h_counts_sorted, statistics.counts_sorted, stream);

  // initialize the category_frequent_index array:
  //   hash table indicating the location of the frequent category in the local 
  //   embedding vector and partial gradient buffers
  std::vector<dtype> h_category_frequent_index(num_categories, num_categories);
  for (size_t i = 0; i < num_frequent; ++i) {
    dtype category = h_categories_sorted[i]
    h_category_frequent_index[category] = (dtype) i;
  }

  // initialize category_location
  //   for each category: global_network_id, local_buffer_index
  std::vector<dtype> category_location(2*num_categories, num_categories);
  size_t indx = 0;
  for (size_t category = 0; category < num_categories; ++category) {
    if (h_category_frequent_index[category] == num_categories) {
      h_category_location[2*category] = indx % num_networks;
      h_category_location[2*category+1] = indx / num_networks;
    }
  }

  // upload the model arrays
  upload_tensor(h_category_frequent_index, category_frequent_index, stream);
  upload_tensor(h_category_location, category_location, stream);

  /// ============================
  /// ============================

}


#include "HugeCTR/include/embeddings/hybrid_embedding_template_defs/hybrid_embedding_model_template_defs.cuh"
}