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

// 
struct CalibrationInitializationData {
  CalibrationInitializationData() {}
  ~CalibrationInitializationData() {}

  // Calibration all-to-all : 
  //   the following two arrays map data sizes to all-to-all times / latencies.
  std::vector<double> all_to_all_data_size; // data size of message per gpu
  std::vector<double> all_to_all_times;     // calibrated all-to-all times

  // Calibration all-reduce : 
  //   the following two arrays map data sizes to all-to-all times / latencies.
  std::vector<double> all_reduce_data_size; // data size of message per gpu
  std::vector<double> all_reduce_times;     // calibrated all-reduce times

  // Alternative calibration: (if no calibration provided)
  //   the threshold for frequent categories is calculated from maximum bandwidths
  //   for the all-reduce and all-to-all respectively. 
  //   This approximation assumes that the communications are bandwidth limited.
  double max_all_reduce_bandwidth; // algorithm bandwidth all-reduce [data size message per gpu in bytes / sec]
  double max_all_to_all_bandwidth; // algorithm bandwidth all-to-all [data size message per gpu in bytes / sec]
};


template <typename dtype>
struct HybridEmbeddingModel {
public:
  HybridEmbeddingModel() {}
  ~HybridEmbeddingModel() {}

  uint32_t node_id;
  uint32_t gpu_id;

  dtype num_frequent;
  dtype num_categories;

  std::vector<uint32_t> num_gpus_per_node; // number of gpus for each node, .size() == number of nodes

  Tensor2<dtype> category_frequent_index;  // is this category a frequent category? => location in cache
  Tensor2<dtype> category_location;        // is this an infrequent category? 
                                           // => location of where the categories are stored
  void init_model(
    const CalibrationInitializationData& calibration,
    Tensor2<dtype> samples
    );
};


template <typename dtype>
struct HybridEmbeddingData {
  std::vector<uint32_t> table_sizes;
  size_t batch_size;
  size_t num_networks;

  // pointer to raw data: iteration from data reader
  Tensor2<dtype> data;
  // flattened data
  Tensor2<dtype> samples;

  cudaStream_t stream;

  // flatten raw input data
  void flatten_samples(cudaStream_t stream);
};
