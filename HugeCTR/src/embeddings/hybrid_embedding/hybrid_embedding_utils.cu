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

namespace HugeCTR {


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
  

// template definitions
#include "HugeCTR/include/embeddings/hybrid_embedding_template_defs/hybrid_embedding_utils_template_defs.cuh"
}