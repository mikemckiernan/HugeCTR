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
#include "HugeCTR/include/embeddings/hybrid_embedding/utils.hpp"
#include "HugeCTR/include/tensor2.hpp"

namespace HugeCTR {

namespace hybrid_embedding {

template <typename dtype>
void download_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, cudaStream_t stream) {
  size_t tensor_size = tensor.get_size_in_bytes() / sizeof(dtype);
  h_tensor.resize(tensor_size);
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  CK_CUDA_THROW_(cudaMemcpyAsync(h_tensor.data(), tensor.get_ptr(), tensor.get_size_in_bytes(),
                                 cudaMemcpyDeviceToHost, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
}

template <typename dtype>
void upload_tensor(std::vector<dtype>& h_tensor, Tensor2<dtype> tensor, cudaStream_t stream) {
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
  CK_CUDA_THROW_(cudaMemcpyAsync(tensor.get_ptr(), h_tensor.data(), h_tensor.size() * sizeof(dtype),
                                 cudaMemcpyHostToDevice, stream));
  CK_CUDA_THROW_(cudaStreamSynchronize(stream));
}

template void download_tensor<uint32_t>(std::vector<uint32_t>& h_tensor, Tensor2<uint32_t> tensor,
                                        cudaStream_t stream);
template void download_tensor<unsigned long>(std::vector<size_t>& h_tensor, Tensor2<size_t> tensor,
                                             cudaStream_t stream);
template void download_tensor<__half>(std::vector<__half>& h_tensor, Tensor2<__half> tensor,
                                      cudaStream_t stream);
template void download_tensor<float>(std::vector<float>& h_tensor, Tensor2<float> tensor,
                                     cudaStream_t stream);
template void upload_tensor<uint32_t>(std::vector<uint32_t>& h_tensor, Tensor2<uint32_t> tensor,
                                      cudaStream_t stream);
template void upload_tensor<unsigned long>(std::vector<size_t>& h_tensor, Tensor2<size_t> tensor,
                                           cudaStream_t stream);
template void upload_tensor<__half>(std::vector<__half>& h_tensor, Tensor2<__half> tensor,
                                    cudaStream_t stream);
template void upload_tensor<float>(std::vector<float>& h_tensor, Tensor2<float> tensor,
                                   cudaStream_t stream);
}  // namespace hybrid_embedding

}  // namespace HugeCTR