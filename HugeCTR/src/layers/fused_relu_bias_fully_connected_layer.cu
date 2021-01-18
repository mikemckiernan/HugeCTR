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

#include <layers/fused_relu_bias_fully_connected_layer.hpp>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

namespace {

template <int BLOCK_WIDTH>
__global__ void reverse_add_bias_and_re_kernel(float* bias, __half* middle, const __half* top,
                                               int ldn) {
  __shared__ __half2 elem[32][BLOCK_WIDTH + 1];
  __shared__ __half2 accu[BLOCK_WIDTH];

  const __half2 zero = TypeFunc<__half2>::zero();

  __half2* middle2 = reinterpret_cast<__half2*>(middle);
  const __half2* top2 = reinterpret_cast<const __half2*>(top);

  int lx, ly, gi;
  int gx_offset = blockIdx.x * BLOCK_WIDTH;
  int gy_offset = blockIdx.y * 32;

  for (int i = 0; i < BLOCK_WIDTH * 32; i += blockDim.x) {
    lx = threadIdx.x % BLOCK_WIDTH;
    ly = (i + threadIdx.x) / BLOCK_WIDTH;
    gi = (ly + gy_offset) * ldn + (lx + gx_offset);

    __half2 t = middle2[gi];
    __half2 mask = __hgt2(t, zero);
    t = __hmul2(__ldg(top2 + gi), mask);

    middle2[gi] = t;
    elem[ly][lx] = t;
  }

  __syncthreads();

  for (int i = 0; i < BLOCK_WIDTH * 32; i += blockDim.x) {
    lx = (i + threadIdx.x) / 32;
    ly = threadIdx.x % 32;

    __half2 val = warpReduceSum(elem[ly][lx]);
    if (ly == 0) {
      accu[lx] = val;
    }
  }

  __syncthreads();

  if (threadIdx.x < BLOCK_WIDTH * 2) {
    __half2 val = accu[threadIdx.x / 2];
    float fval = (threadIdx.x % 2 == 0) ? __low2float(val) : __high2float(val);
    atomicAdd(bias + gx_offset * 2 + threadIdx.x, fval);
  }
}

}  // namespace

FusedReluBiasFullyConnectedLayer::FusedReluBiasFullyConnectedLayer(
    const std::shared_ptr<BufferBlock2<float>>& master_weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_buff,
    const std::shared_ptr<BufferBlock2<__half>>& weights_grad_buff,
    const std::shared_ptr<GeneralBuffer2<CudaAllocator>>& blobs_buff,
    const Tensor2<__half>& train_bottom_tensor_fprop, 
    const Tensor2<__half>& train_bottom_tensor_bprop,
    const Tensor2<__half>& top_tensor_fprop, 
    const Tensor2<__half>& top_tensor_bprop, 
    const std::shared_ptr<GPUResource>& gpu_resource,
    const std::string& pos,
    std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  const auto& bottom_tensor_dim = train_bottom_tensor_fprop.get_dimensions();
  const auto& top_tensor_dim = top_tensor_fprop.get_dimensions();

  if (bottom_tensor_dim.size() != 2 || top_tensor_dim.size() != 2) {
    CK_THROW_(Error_t::WrongInput, "input or output tensor doesn't has two dimensions");
  }

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  if (m % 32 != 0 || n % 64 != 0) {
    CK_THROW_(Error_t::WrongInput,
              "The first dimension of bottom tensor must be a multiple of 32, the second dimension "
              "of top tensor must be a multiple of 64.");
  }

  std::vector<size_t> kernel_dim = {k, n};
  std::vector<size_t> bias_dim = {1, n};

  {
    Tensor2<float> tensor;
    master_weights_buff->reserve(kernel_dim, &tensor);
    weights_.push_back(tensor);
  }
  {
    Tensor2<float> tensor;
    master_weights_buff->reserve(bias_dim, &tensor);
    weights_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_buff->reserve(kernel_dim, &tensor);
    weights_half_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_buff->reserve(bias_dim, &tensor);
    weights_half_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_grad_buff->reserve(kernel_dim, &tensor);
    weights_grad_.push_back(tensor);
  }
  {
    Tensor2<__half> tensor;
    weights_grad_buff->reserve(bias_dim, &tensor);
    weights_grad_.push_back(tensor);
  }

  if(pos=="Head")
    pos_ = HEAD;
  else if(pos=="Body")
    pos_ = BODY;
  else if(pos=="Tail")
    pos_ = TAIL;
  else if(pos=="Isolated")
    pos_ = ISOLATED;
  train_bottom_tensor_fprop_ = train_bottom_tensor_fprop;
  if(pos_ == HEAD || pos_ == ISOLATED)
    train_bottom_tensor_bprop_ = train_bottom_tensor_fprop;
  else
    train_bottom_tensor_bprop_ = train_bottom_tensor_bprop;
  top_tensor_fprop_ = top_tensor_fprop;
  top_tensor_bprop_ = top_tensor_bprop;
  blobs_buff->reserve(bias_dim, &bias_grad_tensor_);

}

void FusedReluBiasFullyConnectedLayer::initialize() {

  // TODO: We need different bottom desc based on is_train or not
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_fprop_.get_dimensions();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  CK_CUBLAS_THROW_(cublasLtMatmulDescCreate(&cublas_op_desc_, CUBLAS_COMPUTE_16F, CUDA_R_16F));

  cublasOperation_t trans  = CUBLAS_OP_N;
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));
  cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_BIAS;
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
  const __half* bias = weights_half_[1].get_ptr();
  CK_CUBLAS_THROW_(cublasLtMatmulDescSetAttribute(cublas_op_desc_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));


  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_kernel_desc_, CUDA_R_16F, n, k, n));
  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_bottom_desc_, CUDA_R_16F, k, m, k));
  CK_CUBLAS_THROW_(cublasLtMatrixLayoutCreate(&cublas_top_desc_, CUDA_R_16F, n, m, n));

  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceCreate(&cublas_preference_));

  cublaslt_workspace_size_ = 1024*1024*8; // Set it to 8MB for now
  CK_CUDA_THROW_(cudaMalloc(&cublaslt_workspace_, cublaslt_workspace_size_));
  CK_CUBLAS_THROW_(cublasLtMatmulPreferenceSetAttribute(cublas_preference_, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size_, sizeof(cublaslt_workspace_size_)));

  // By default set algo to best estimated heurstic
  cublasLtMatmulHeuristicResult_t heuristic_result;
  int returned_res = 0;
  CK_CUBLAS_THROW_(cublasLtMatmulAlgoGetHeuristic(get_gpu().get_cublaslt_handle(), cublas_op_desc_, cublas_kernel_desc_, cublas_bottom_desc_, cublas_top_desc_, cublas_top_desc_, cublas_preference_ ,1, &heuristic_result, &returned_res));

  memcpy(&falgo_k_, &heuristic_result.algo, sizeof(falgo_k_));

  if (returned_res == 0) {
    CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }
}

void FusedReluBiasFullyConnectedLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = weights_half_[0].get_ptr();
  const __half* bias = weights_half_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(is_train).get_ptr();
  __half* top_fprop = top_tensor_fprop_.get_ptr();
  __half* top_bprop = top_tensor_bprop_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(is_train).get_dimensions();
  const auto& top_tensor_dim = top_tensor_fprop_.get_dimensions();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  const __half alpha = 1.0f;
  const __half beta = 0.0f;

  CK_CUBLAS_THROW_(cublasLtMatmul(get_gpu().get_cublaslt_handle(),
              cublas_op_desc_,
              &alpha,
              kernel,
              cublas_kernel_desc_,
              bottom,
              cublas_bottom_desc_,
              &beta,
              top_fprop,
              cublas_top_desc_,
              top_fprop,
              cublas_top_desc_,
              &falgo_k_,
              cublaslt_workspace_,
              cublaslt_workspace_size_,
              get_gpu().get_stream()));

    if(pos_ == TAIL || pos_ == ISOLATED)
    {
        size_t len = top_tensor_fprop_.get_num_elements();
        CK_CUDA_THROW_(cudaMemcpyAsync(top_bprop, top_fprop,
            len*sizeof(__half), cudaMemcpyDeviceToDevice, get_gpu().get_stream()));
    }
#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif
}

void FusedReluBiasFullyConnectedLayer::bprop() {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = weights_half_[0].get_ptr();
  const __half* top = top_tensor_bprop_.get_ptr();
  __half* middle = top_tensor_fprop_.get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* bottom_bprop = get_bottom_tensor_bprop(true).get_ptr();
  float* bias_grad_float = bias_grad_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_bprop(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_bprop_.get_dimensions();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  const float alpha = 1.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  initialize_array<<<(n - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(bias_grad_float, n,
                                                                            0.0f);

  dim3 blocks(n / 64, m / 32);
  reverse_add_bias_and_re_kernel<32>
      <<<blocks, 512, 0, get_gpu().get_stream()>>>(bias_grad_float, middle, top, n / 2);

  convert_array<<<(n - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(bias_grad, bias_grad_float,
                                                                         n);

  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                                &alpha, middle, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k,
                                kernel_grad, CUDA_R_16F, n, CUDA_R_32F, balgo_k_));

  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                                &alpha, kernel, CUDA_R_16F, n, middle, CUDA_R_16F, n, &beta_x,
                                bottom_bprop, CUDA_R_16F, k, CUDA_R_32F, balgo_x_));

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif

}

void FusedReluBiasFullyConnectedLayer::search_algorithm() {

  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());
  const size_t repeat_num = 100;
  const int max_algo_count = 16;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* top = top_tensor_fprop_.get_ptr();
  __half* kernel = weights_half_[0].get_ptr();
  __half* bias = weights_half_[1].get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = top_tensor_fprop_.get_dimensions();

  size_t m = bottom_tensor_dim[0];
  size_t n = top_tensor_dim[1];
  size_t k = bottom_tensor_dim[1];

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  cudaEvent_t start, stop;
  CK_CUDA_THROW_(cudaEventCreate(&start));
  CK_CUDA_THROW_(cudaEventCreate(&stop));

  cublasLtMatmulHeuristicResult_t heuristic_result[max_algo_count] = {0};
  int algo_count = 0;
  CK_CUBLAS_THROW_(cublasLtMatmulAlgoGetHeuristic(get_gpu().get_cublaslt_handle(), cublas_op_desc_, cublas_kernel_desc_, cublas_bottom_desc_, cublas_top_desc_, cublas_top_desc_, cublas_preference_, 1, heuristic_result, &algo_count));

  if (algo_count == 0) {
      CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  for (int algoIdx = 0; algoIdx < algo_count; algoIdx++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
        status = cublasLtMatmul(get_gpu().get_cublaslt_handle(),
                cublas_op_desc_,
                &alpha,
                kernel,
                cublas_kernel_desc_,
                bottom,
                cublas_bottom_desc_,
                &beta,
                top,
                cublas_top_desc_,
                top,
                cublas_top_desc_,
                &heuristic_result[algoIdx].algo,
                cublaslt_workspace_,
                cublaslt_workspace_size_,
                get_gpu().get_stream());
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));

    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for fprop, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      memcpy(&falgo_k_, &heuristic_result[algoIdx].algo, sizeof(falgo_k_));
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Start, end for search
  const cublasGemmAlgo_t startAlgo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  const cublasGemmAlgo_t endAlgo = CUBLAS_GEMM_ALGO15_TENSOR_OP;

  // Search all the algorithm for balgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                            &alpha, top, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta, kernel_grad,
                            CUDA_R_16F, n, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }
    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for bprop_W, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_k_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_x_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    const __half alpha = 1.0f;
    const __half beta = 0.0f;

    // Record start event
    CK_CUDA_THROW_(cudaEventRecord(start, get_gpu().get_stream()));
    for (size_t i = 0; i < repeat_num && status == CUBLAS_STATUS_SUCCESS; ++i) {
      status = cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                            &alpha, kernel, CUDA_R_16F, n, top, CUDA_R_16F, n, &beta, bottom,
                            CUDA_R_16F, k, CUDA_R_32F, static_cast<cublasGemmAlgo_t>(testAlgo));
    }

    CK_CUDA_THROW_(cudaEventRecord(stop, get_gpu().get_stream()));
    CK_CUDA_THROW_(cudaEventSynchronize(stop));
    CK_CUDA_THROW_(cudaEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this alorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != CUBLAS_STATUS_SUCCESS) {
      //      printf("The algorithms %d is not supported for bprop_Xn, skipped.\n", testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<cublasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // printf("The algorithm selection for falgo_k_, balgo_k_, balgo_x_ are: %d, %d and %d.\n",
  //        (int)falgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_k_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP,
  //        (int)balgo_x_ - CUBLAS_GEMM_DEFAULT_TENSOR_OP);

  // Output msg
  // MESSAGE_("The fully-connected layer has finished choosing the algorithm for cublas Gemm.");
  // Clean-up
  CK_CUDA_THROW_(cudaEventDestroy(start));
  CK_CUDA_THROW_(cudaEventDestroy(stop));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_uniform_initializer(const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = top_tensor_fprop_.get_dimensions()[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_xavier_uniform_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = top_tensor_fprop_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_xavier_norm_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = top_tensor_fprop_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_default_initializer(const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = top_tensor_fprop_.get_dimensions()[1];

  std::unique_ptr<DataSimulator> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator(1.f, data_simu::Mode_t::Fan_avg,
                                            data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else if (1 == index) {
    float stddev = sqrt(1.f / top_dim);
    simu.reset(new GaussianDataSimulator(0, stddev, -2 * stddev, 2 * stddev));
  } else {
    CK_THROW_(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

}  // namespace HugeCTR
  