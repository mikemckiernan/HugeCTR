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

#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "HugeCTR/include/common.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/model.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

/// init_model calculates the optimal number of frequent categories
/// given the calibration of the all-to-all and all-reduce.
template <typename dtype>
void Model<dtype>::init_model(const CommunicationType communication_type,
                              const CalibrationData &calibration, const Data<dtype> &data,
                              cudaStream_t stream) {
  // Requires:
  //     communication_type
  //     calibration
  //     number of iterations to be read into data
  //           here: data.table_sizes, data.samples, data.num_networks
  // calculate the total number of categories

  num_categories = (dtype)0;
  for (size_t i = 0; i < data.table_sizes.size(); ++i) num_categories += data.table_sizes[i];
  num_networks = 0;
  for (size_t n = 0; n < h_num_networks_per_node.size(); ++n)
    num_networks += h_num_networks_per_node[n];

  // determine the number of frequent categories

  // list the top categories sorted by count
  Tensor2<dtype> &samples = data.samples;
  Statistics<dtype> statistics(samples.get_size_in_bytes() / sizeof(dtype));
  statistics.sort_categories_by_count(samples, stream);

  // from the sorted count, determine the number of frequent categories
  //
  // If the calibration data is present, this is used to calculate the number
  // of frequent categories.  Otherwise use the threshold required by the
  // communication type.
  num_frequent = ModelInitializationFunctors<dtype>::calculate_num_frequent_categories(
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
    dtype category = h_categories_sorted[i];
    h_category_frequent_index[category] = (dtype)i;
  }

  // initialize category_location
  //   for each category: global_network_id, local_buffer_index
  std::vector<dtype> h_category_location(2 * num_categories, num_categories);
  size_t indx = 0;
  for (size_t category = 0; category < num_categories; ++category) {
    if (h_category_frequent_index[category] == num_categories) {
      h_category_location[2 * category] = indx % num_networks;
      h_category_location[2 * category + 1] = indx / num_networks;
    }
  }

  // upload the model arrays
  upload_tensor(h_category_frequent_index, category_frequent_index, stream);
  upload_tensor(h_category_location, category_location, stream);

  /// ============================
  /// ============================
}

template class Data<uint32_t>;
template class Data<unsigned long>;
}  // namespace hybrid_embedding

}  // namespace HugeCTR