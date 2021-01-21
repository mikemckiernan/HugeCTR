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

#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_calibration.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_data.hpp"
#include "HugeCTR/include/embeddings/hybrid_embedding/hybrid_embedding_utils.hpp"
  
#include <algorithm>
#include <iostream>
#include <vector>


namespace HugeCTR {


///
/// interpolate data_size using the two calibration data 
///   calibrated_data_size, calibrated_times
///   return communication_times
///
float CalibrationData::interpolate(
    const Tensor2<float> &calibrated_data_size,
    const Tensor2<float> &calibrated_times,
    const Tensor2<float> &data_size,
    Tensor2<float> &communication_times
) {
  // TODO: implement this
}


///
/// Convenience function for interpolating all-to-all communication times from 
/// calibrated data
///
template <typename dtype>
float CalibrationData::interpolate_all_reduce(
  const Tensor2<float> &data_size,
  Tensor2<float> &communication_times
) {
  interpolate(
    all_reduce_data_size, all_reduce_times, data_size, communication_times
  );
}


///
/// Convenience function for interpolating all-to-all communication times from 
/// calibrated data
///
template <typename dtype>
float CalibrationData::interpolate_all_to_all(
  const Tensor2<float> &data_size,
  Tensor2<float> &communication_times
) {
  interpolate(
    all_to_all_data_size, all_to_all_times, data_size, communication_times
  );
}


/// 
/// Calculate the number of frequent categories from data
///
template <typename dtype>
uint32_t CalculateNumberOfFrequentCategories(
  CalibrationData<dtype> calibration_data,
  HybridEmbeddingStatistics<dtype> statistics
);


// Calculate threshold such that for the worst case distribution there will 
// be one duplication per network on average
float CalibrationData::calculate_threshold(
  CommunicationType communication_type,
  size_t batch_size, 
  size_t num_networks,
  size_t num_iterations,
  size_t num_tables
) {
  float count_threshold = 1.f;
  
  // for NVLink capture effectively all duplications with number of categories
  double M = (double) batch_size / (double) num_networks;
  double p_dup_max = 1.0 / M;
  double count_threshold = 1.;
  switch(communication_type) {
  case IB_NVLink:
    count_threshold = (double) num_nodes * all_to_all_bandwidth / all_reduce_bandwidth 
                    * (double) num_nodes / ((double) num_nodes - 1.);
    break;
  case NVLink:
    // count threshold such that the probability of duplication is less than p_dup_max
    //   even if there are batch size number of categories that occur more often,
    //   there will be a duplication at most once every iteration per gpu
    //
    // d_duplication(category) \approx 1/2 M (M-1) \left( \frac{count}{batch_sizexnum_iterations} \right)^2
    count_threshold = (float) ((double) batch_size * (double) num_iterations * (double) num_tables 
                    * sqrt(2.0 * p_dup_max / (M *(M-1))));
    break;
  default:
    CK_THROW(Error_t::WrongInput, "Unknown communication type, expecting IB_NVLink or NVLink");
  }

  return count_threshold;
}


#include "HugeCTR/include/embeddings/hybrid_embedding_includes/hybrid_embedding_calibration_includes.cuh"
}
