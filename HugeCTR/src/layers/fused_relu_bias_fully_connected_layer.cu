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

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/kernel/default_gemm_with_reduction.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/epilogue/thread/linear_combination_drelu.h"
#include "cutlass/epilogue/thread/linear_combination_dgelu.h"

// // #include "../../common/cutlass_unit_test.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"
#include "layers/gemm_with_reduction.hpp"
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
    const Tensor2<__half>& train_in_tensor, 
    const Tensor2<__half>& mask_in_tensor,
    const Tensor2<__half>& dRelu_in_tensor,
    const Tensor2<float>& db_in_tensor,   
    const Tensor2<__half>& train_out_tensor, 
    const Tensor2<__half>& mask_out_tensor, 
    const Tensor2<__half>& dRelu_out_tensor,
    const Tensor2<float>& db_out_tensor,
    const std::shared_ptr<GPUResource>& gpu_resource,
    const std::string& pos,
    std::vector<Initializer_t> initializer_types)
    : Layer(gpu_resource, initializer_types),
      balgo_k_(CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      balgo_x_(CUBLAS_GEMM_DEFAULT_TENSOR_OP) {
  const auto& bottom_tensor_dim = train_in_tensor.get_dimensions();
  const auto& top_tensor_dim = train_out_tensor.get_dimensions();

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
  train_in_tensor_ = train_in_tensor;
  if(pos_ == HEAD || pos_ == ISOLATED)
    mask_in_tensor_ = train_in_tensor;
  else {
    mask_in_tensor_  = mask_in_tensor;
    dRelu_in_tensor_ = dRelu_in_tensor;
    db_in_tensor_    = db_in_tensor;
  }
  train_out_tensor_  = train_out_tensor;
  mask_out_tensor_   = mask_out_tensor;
  dRelu_out_tensor_  = dRelu_out_tensor;
  db_out_tensor_     = db_out_tensor;
  blobs_buff->reserve(bias_dim, &bias_grad_tensor_);

}

void FusedReluBiasFullyConnectedLayer::initialize() {

  // TODO: We need different bottom desc based on is_train or not
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

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

  if(pos_ == BODY || pos_ == TAIL) gemm_dRelu_bgrad_init();

  if (returned_res == 0) {
    CK_CUBLAS_THROW_(CUBLAS_STATUS_NOT_SUPPORTED);
  }
}

void FusedReluBiasFullyConnectedLayer::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = weights_half_[0].get_ptr();
  const __half* bias = weights_half_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(is_train).get_ptr();
  __half* top_fprop = train_out_tensor_.get_ptr();
  __half* top_bprop = mask_out_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(is_train).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

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
        size_t len = train_out_tensor_.get_num_elements();
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
  const __half* top = mask_out_tensor_.get_ptr();
  __half* middle = train_out_tensor_.get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* bottom_bprop = get_bottom_tensor_bprop(true).get_ptr();
  float* bias_grad_float = bias_grad_tensor_.get_ptr();
  __half* dRelu_top    = dRelu_out_tensor_.get_ptr();    
  float* db_top    = db_out_tensor_.get_ptr();    

  const auto& bottom_tensor_dim = get_bottom_tensor_bprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

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
  // if(pos_ == BODY || pos_ == HEAD) {  
  //   __half* db_top_host = new __half[weights_grad_[1].get_num_elements()];
  //   cudaMemcpyAsync(db_top_host, db_top, weights_grad_[1].get_num_elements()*sizeof(__half),
  //       cudaMemcpyDeviceToHost, get_gpu().get_stream());
  //   __half* db_host = new __half[weights_grad_[1].get_num_elements()];
  //   cudaMemcpyAsync(db_host, bias_grad, weights_grad_[1].get_num_elements()*sizeof(__half),
  //       cudaMemcpyDeviceToHost, get_gpu().get_stream());
  //   printf("%d, %d, %d\n", m, n, k);
  //   for (unsigned i = 0; i < weights_grad_[1].get_num_elements(); i++)
  //   {
  //     if(abs(float(db_host[i]) - float(db_top_host[i]))>1e-3)
  //     {
  //       printf("%d, %f, %f\n", i, (float)db_host[i], (float)db_top_host[i]);
  //       // exit(-1);
  //     }
  //   }
  //   printf("Pass\n");
  // }

  // if(pos_ == BODY || pos_ == HEAD) {
  //   __half* dRelu_top_host = new __half[dRelu_out_tensor_.get_num_elements()];
  //   cudaMemcpyAsync(dRelu_top_host, dRelu_top, dRelu_out_tensor_.get_num_elements()*sizeof(__half),
  //       cudaMemcpyDeviceToHost, get_gpu().get_stream());
  //   __half* middle_host = new __half[dRelu_out_tensor_.get_num_elements()];
  //   cudaMemcpyAsync(middle_host, middle, dRelu_out_tensor_.get_num_elements()*sizeof(__half),
  //       cudaMemcpyDeviceToHost, get_gpu().get_stream());
  //   printf("%d, %d, %d\n", m, n, k);
  //   for (unsigned i = 0; i < dRelu_out_tensor_.get_num_elements(); i++)
  //   {
  //     if(abs(float(middle_host[i]) - float(dRelu_top_host[i]))>1e-4)
  //     {
  //       printf("%d, %f, %f\n", i, (float)middle_host[i], (float)dRelu_top_host[i]);
  //       exit(-1);
  //     }
  //   }
  //   printf("Pass\n");
  // }

  if(pos_ == BODY || pos_ == HEAD) {
    middle = dRelu_out_tensor_.get_ptr();
    bias_grad = (__half*)db_out_tensor_.get_ptr();
  }
      // cudaMemcpyAsync(middle, dRelu_top, dRelu_out_tensor_.get_num_elements()*sizeof(__half),
        // cudaMemcpyDeviceToDevice, get_gpu().get_stream());
  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
                                &alpha, middle, CUDA_R_16F, n, bottom, CUDA_R_16F, k, &beta_k,
                                kernel_grad, CUDA_R_16F, n, CUDA_R_32F, balgo_k_));

  CK_CUBLAS_THROW_(cublasGemmEx(get_gpu().get_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
                                &alpha, kernel, CUDA_R_16F, n, middle, CUDA_R_16F, n, &beta_x,
                                bottom_bprop, CUDA_R_16F, k, CUDA_R_32F, balgo_x_));
  if(pos_ == BODY || pos_ == TAIL) {
    gemm_dRelu_bgrad_run();
    // float* db    = (float*)db_in_tensor_.get_ptr();    
    // float* db_host = new float[k];
    // cudaMemcpyAsync(db_host, db, k*sizeof(float), cudaMemcpyDeviceToHost, get_gpu().get_stream());

    // __half* dRelu_bottom = dRelu_in_tensor_.get_ptr();
    // __half* bottom_ptr ;
    // cudaMalloc((void**)&bottom_ptr, m*k*sizeof(__half));
    // CK_CUDA_THROW_(cudaMemcpyAsync(bottom_ptr, bottom,
    //     m*k*sizeof(__half), cudaMemcpyDeviceToDevice, get_gpu().get_stream()));    
    // reverse_add_bias_and_re_kernel<32>
    //   <<<blocks, 512, 0, get_gpu().get_stream()>>>(bias_grad_float, bottom_ptr, bottom_bprop, n / 2);
    // float* db_ptr_host = new float[k];
    // cudaMemcpyAsync(db_ptr_host, bias_grad_float, k*sizeof(float), cudaMemcpyDeviceToHost, get_gpu().get_stream());   
    // for (int i = 0; i < k; i++)
    // {
    //   printf("%d, %f, %f\n", i, db_ptr_host[i], db_host[i]);
    // }            

    // __half* dRelu_bottom_host = new __half[dRelu_in_tensor_.get_num_elements()];
    // cudaMemcpyAsync(dRelu_bottom_host, dRelu_bottom, dRelu_in_tensor_.get_num_elements()*sizeof(__half),
    //     cudaMemcpyDeviceToHost, get_gpu().get_stream());
    // for (int i = 0; i < m; i++)
    // {
    //   printf("%d, %f, %f\n", i, (float)dRelu_bottom_host[i], (float)db_host[i]);
    // }     
    // __half* bottom_ptr_host = new __half[dRelu_in_tensor_.get_num_elements()];
    // cudaMemcpyAsync(bottom_ptr_host, bottom_ptr, dRelu_in_tensor_.get_num_elements()*sizeof(__half),
    //     cudaMemcpyDeviceToHost, get_gpu().get_stream());      
    // printf("%d, %d, %d\n", m, n, k);
    // for (int i = 0; i < m*n; i++)
    // {
    //   if(abs(float(bottom_ptr_host[i]) - float(dRelu_bottom_host[i]))>1e-5)
    //   {
    //     printf("%d, %f, %f\n", i, (float)bottom_ptr_host[i], (float)dRelu_bottom_host[i]);
    //     exit(-1);
    //   }
    // }
    // printf("Pass\n");
  }

#ifndef NDEBUG
  cudaDeviceSynchronize();
  CK_CUDA_THROW_(cudaGetLastError());
#endif

}

void FusedReluBiasFullyConnectedLayer::gemm_dRelu_bgrad_init()
{
  const __half* kernel = weights_half_[0].get_ptr();
  const __half* top = mask_out_tensor_.get_ptr();
  __half* middle = train_out_tensor_.get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* bottom_bprop = get_bottom_tensor_bprop(true).get_ptr();
  float* bias_grad_float = bias_grad_tensor_.get_ptr();
  __half* dRelu_bottom = dRelu_in_tensor_.get_ptr();
  float* db_bottom = db_in_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_bprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  // cutlassGemmWithReduction<Gemm>(
  //   W, dRelu_top, dRelu_bottom, db, mask,
  //   {k, m, n},
  //   cutlass::gemm::GemmUniversalMode::kGemm,
  //   1,
  //   float(1),
  //   float(0)
  // );
  bprop_fusion_ = new TestbedGemmWithReduction<Gemm>(); 
  reinterpret_cast<TestbedGemmWithReduction<Gemm>*>(bprop_fusion_)->initialize(
    kernel, middle, dRelu_bottom, db_bottom, bottom,
    cutlass::gemm::GemmUniversalMode::kGemm,
    {k, m, n},
    1,
    float(1),
    float(0),
    get_gpu().get_stream());  
}

void FusedReluBiasFullyConnectedLayer::gemm_dRelu_bgrad_run()
{
  const __half* kernel = weights_half_[0].get_ptr();
  const __half* top = mask_out_tensor_.get_ptr();
  __half* middle = train_out_tensor_.get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();
  const __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* bottom_bprop = get_bottom_tensor_bprop(true).get_ptr();
  float* bias_grad_float = bias_grad_tensor_.get_ptr();
  __half* dRelu_bottom = dRelu_in_tensor_.get_ptr();
  float* db_bottom = db_in_tensor_.get_ptr();

  const auto& bottom_tensor_dim = get_bottom_tensor_bprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

  int m = bottom_tensor_dim[0];
  int n = top_tensor_dim[1];
  int k = bottom_tensor_dim[1];

  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationDRelu<
    float,
    float,
    cutlass::half_t,
    cutlass::half_t,
    8
  >;

  using GemmKernel = 
    typename cutlass::gemm::kernel::DefaultGemmWithReduction<
      cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
      cutlass::half_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
      cutlass::half_t, cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 8>,
      EpilogueOutputOp,
      cutlass::plus<float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
      2,
      cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  reinterpret_cast<TestbedGemmWithReduction<Gemm>*>(bprop_fusion_)->run(
    db_bottom,
    get_gpu().get_stream()
  );
}

void FusedReluBiasFullyConnectedLayer::search_algorithm() {

  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());
  const size_t repeat_num = 100;
  const int max_algo_count = 16;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor_fprop(true).get_ptr();
  __half* top = train_out_tensor_.get_ptr();
  __half* kernel = weights_half_[0].get_ptr();
  __half* bias = weights_half_[1].get_ptr();
  __half* kernel_grad = weights_grad_[0].get_ptr();
  __half* bias_grad = weights_grad_[1].get_ptr();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor_fprop(true).get_dimensions();
  const auto& top_tensor_dim = train_out_tensor_.get_dimensions();

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
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_xavier_uniform_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_xavier_norm_initializer(
    const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FusedReluBiasFullyConnectedLayer::get_default_initializer(const int index) {
  size_t bottom_dim = get_bottom_tensor_fprop(true).get_dimensions()[1];
  size_t top_dim = train_out_tensor_.get_dimensions()[1];

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
  
