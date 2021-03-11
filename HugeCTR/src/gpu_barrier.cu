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

#include <gpu_barrier.hpp>
#include <utils.cuh>

namespace HugeCTR
{
  namespace gpu_barrier
  {
    __global__ void sync_all_gpus_cuda(size_t** d_rem_barrier_flags, 
        size_t my_local_id, size_t ndevs)
    {
      size_t count = d_rem_barrier_flags[my_local_id][my_local_id];
      size_t g_tid = blockIdx.x*blockDim.x + threadIdx.x;
      if (g_tid < ndevs) {
        volatile size_t* rem_flag = d_rem_barrier_flags[g_tid];
        volatile size_t* my_flag = d_rem_barrier_flags[my_local_id];
        rem_flag[my_local_id] = (count + 1);
        while(my_flag[g_tid] < (count + 1)) {}
      }
      __syncthreads();
    }

    // Only single CTA launch
    __global__ void sync_all_gpus_report_host_cuda(size_t** d_rem_barrier_flags, 
        size_t* d_report_count, size_t* h_report_ptr, size_t my_local_id, size_t ndevs)
    {
      size_t count = d_rem_barrier_flags[my_local_id][my_local_id];
      size_t g_tid = blockIdx.x*blockDim.x + threadIdx.x;
      if (g_tid < ndevs) {
        volatile size_t* rem_flag = d_rem_barrier_flags[g_tid];
        volatile size_t* my_flag = d_rem_barrier_flags[my_local_id];
        rem_flag[my_local_id] = (count + 1);
        while(my_flag[g_tid] < (count + 1)) {}
      }
      __syncthreads();
      if ((g_tid == 0) && (my_local_id == 0))
      {
        *h_report_ptr = *d_report_count;
      }
    }
  }

  using namespace gpu_barrier;
  
  GPUBarrier::GPUBarrier(const std::shared_ptr<const ResourceManager>& resource_manager):
    resource_manager_(resource_manager)
  {
    num_gpus_ = resource_manager_->get_local_gpu_count();
    d_barrier_flags_ = new size_t*[num_gpus_];
    d_rem_barrier_flags_ = new size_t**[num_gpus_];
    auto& dev_list = resource_manager_->get_local_gpu_device_id_list();

    for (size_t g = 0; g < num_gpus_; g++) {
      CK_CUDA_THROW_(cudaSetDevice(dev_list[g]));
      CK_CUDA_THROW_(cudaMalloc(&d_barrier_flags_[g], sizeof(size_t)));
    }

    for (size_t g = 0; g < num_gpus_; g++) {
      CK_CUDA_THROW_(cudaSetDevice(dev_list[g]));
      CK_CUDA_THROW_(cudaMalloc(&d_rem_barrier_flags_[g], num_gpus_*sizeof(size_t*)));
      CK_CUDA_THROW_(cudaMemcpy(d_rem_barrier_flags_[g], d_barrier_flags_, num_gpus_*sizeof(size_t*), cudaMemcpyHostToDevice));
    }
  }

  void GPUBarrier::sync_all_gpus(const cudaStream_t* streams)
  {
    auto& dev_list = resource_manager_->get_local_gpu_device_id_list();
    constexpr size_t MAX_TPB = 256;
    size_t n_blocks = ceildiv<size_t>(num_gpus_, MAX_TPB);
    for (size_t g = 0; g < num_gpus_; g++) {
      CK_CUDA_THROW_(cudaSetDevice(dev_list[g]));
      sync_all_gpus_cuda<<<n_blocks, MAX_TPB, 0, streams[g]>>>(d_rem_barrier_flags_[g], g, num_gpus_);
    }
  }

  void GPUBarrier::sync_all_gpus(const cudaStream_t stream, size_t device_id)
  {
    auto& dev_list = resource_manager_->get_local_gpu_device_id_list();
    constexpr size_t MAX_TPB = 256;
    size_t n_blocks = ceildiv<size_t>(num_gpus_, MAX_TPB);
    CK_CUDA_THROW_(cudaSetDevice(dev_list[device_id]));
    sync_all_gpus_cuda<<<n_blocks, MAX_TPB, 0, stream>>>(d_rem_barrier_flags_[device_id], device_id, num_gpus_);
  }

  void GPUBarrier::sync_all_gpus_report_host(size_t** d_report_count,
      size_t* h_report_ptr, const cudaStream_t* streams)
  {
    auto& dev_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++)
    {
      CK_CUDA_THROW_(cudaSetDevice(dev_list[g]));
      sync_all_gpus_report_host_cuda<<<1, num_gpus_, 0, streams[g]>>>(
          d_rem_barrier_flags_[g], d_report_count[g], h_report_ptr, g, num_gpus_);
    }
  }
  
  void GPUBarrier::sync_all_gpus_report_host(size_t* d_report_count,
      size_t* h_report_ptr, const cudaStream_t stream, size_t device_id)
  {
    auto& dev_list = resource_manager_->get_local_gpu_device_id_list();
    CK_CUDA_THROW_(cudaSetDevice(dev_list[device_id]));
    sync_all_gpus_report_host_cuda<<<1, num_gpus_, 0, stream>>>(
        d_rem_barrier_flags_[device_id], d_report_count, h_report_ptr, device_id, num_gpus_);
  }

  GPUBarrier::~GPUBarrier()
  {
    auto& dev_list = resource_manager_->get_local_gpu_device_id_list();
    for (size_t g = 0; g < num_gpus_; g++) {
      cudaSetDevice(dev_list[g]);
      cudaFree(&d_rem_barrier_flags_[g]);
      cudaFree(&d_barrier_flags_[g]);
    }
    delete d_rem_barrier_flags_;
    delete d_barrier_flags_;
  }
}
