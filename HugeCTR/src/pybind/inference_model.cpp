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
#include <omp.h>
#include <tqdm.h>

#include <pybind/inference_model.hpp>

namespace HugeCTR {

InferenceModel::InferenceModel(const std::string& model_config_path,
                               const InferenceParams& inference_params)
    : inference_params_(inference_params),
      inference_parser_(read_json_file(model_config_path)),
      resource_manager_(ResourceManagerCore::create({inference_params.deployed_devices}, 0)),
      global_max_batch_size_(inference_params_.max_batchsize) {
  HCTR_CHECK_HINT(resource_manager_->get_local_gpu_count() > 0, "deployed_devices cannot be empty");
  HCTR_CHECK_HINT(global_max_batch_size_ % resource_manager_->get_local_gpu_count() == 0,
                  "max_batchsize should be divisible by the number of deployed_devices");
  inference_params_.max_batchsize =
      global_max_batch_size_ / resource_manager_->get_local_gpu_count();
  std::vector<std::string> model_config_path_array{model_config_path};
  std::vector<InferenceParams> inference_params_array{inference_params_};
  parameter_server_config ps_config{model_config_path_array, inference_params_array};
  parameter_server_ = HierParameterServerBase::create(ps_config, inference_params_array);

  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    inference_params_.device_id = resource_manager_->get_local_gpu(i)->get_device_id();
    CudaDeviceContext context(inference_params_.device_id);
    auto embedding_cache = parameter_server_->get_embedding_cache(inference_params_.model_name,
                                                                  inference_params_.device_id);
    inference_sessions_.emplace_back(
        new InferenceSession(model_config_path, inference_params_, embedding_cache));
  }

  inference_params_.max_batchsize = global_max_batch_size_;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaAllocator>>> buffs;
  std::vector<std::shared_ptr<GeneralBuffer2<CudaHostAllocator>>> host_buffs;
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); ++i) {
    buffs.push_back(GeneralBuffer2<CudaAllocator>::create());
    host_buffs.push_back(GeneralBuffer2<CudaHostAllocator>::create());
    pred_tensor_list_.push_back(Tensor2<float>());
    label_tensor_list_.push_back(Tensor2<float>());
    dense_tensor_list_.push_back(Tensor2<float>());
    key_tensor_list_64_.push_back(Tensor2<long long>());
    key_tensor_list_32_.push_back(Tensor2<unsigned int>());
    rowoffset_tensor_list_.push_back(Tensor2<int>());
  }

  size_t batch_size_per_gpu = global_max_batch_size_ / resource_manager_->get_local_gpu_count();
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); ++i) {
    CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
    auto& buff = buffs[i];
    auto& host_buff = host_buffs[i];
    buff->reserve({batch_size_per_gpu, inference_parser_.label_dim}, &pred_tensor_list_[i]);
    buff->reserve({batch_size_per_gpu, inference_parser_.label_dim}, &label_tensor_list_[i]);
    buff->reserve({batch_size_per_gpu, inference_parser_.dense_dim}, &dense_tensor_list_[i]);
    buff->reserve(
        {batch_size_per_gpu * inference_parser_.slot_num + inference_parser_.num_embedding_tables},
        &rowoffset_tensor_list_[i]);
    if (inference_params_.i64_input_key) {
      host_buff->reserve({batch_size_per_gpu * inference_parser_.max_feature_num_per_sample},
                         &key_tensor_list_64_[i]);
    } else {
      host_buff->reserve({batch_size_per_gpu * inference_parser_.max_feature_num_per_sample},
                         &key_tensor_list_32_[i]);
    }
    buff->allocate();
    host_buff->allocate();
  }
}

InferenceModel::~InferenceModel() {
  for (auto device : resource_manager_->get_local_gpu_device_id_list()) {
    CudaDeviceContext context(device);
    cudaDeviceSynchronize();
  }
}

void InferenceModel::reset_reader_tensor_list() {
  reader_label_tensor_list_.clear();
  reader_dense_tensor_list_.clear();
  sparse_input_map_32_.clear();
  sparse_input_map_64_.clear();
}

void InferenceModel::predict(float* pred_output, const size_t num_batches,
                             const std::string& source, const DataReaderType_t data_reader_type,
                             const Check_t check_type,
                             const std::vector<long long>& slot_size_array) {
  reset_reader_tensor_list();
  // TODO: currently data reader can only be created on a single GPU to read the file list that may
  // contain only one parquet/binary file
  std::shared_ptr<ResourceManager> reader_resource_manager{
      ResourceManagerCore::create({{inference_params_.deployed_devices[0]}}, 0)};
  if (inference_params_.i64_input_key) {
    create_datareader<long long>()(
        inference_params_, inference_parser_, data_reader_, reader_resource_manager,
        sparse_input_map_64_, reader_label_tensor_list_, reader_dense_tensor_list_, source,
        data_reader_type, check_type, slot_size_array, true);  // repeat dataset
  } else {
    create_datareader<unsigned int>()(
        inference_params_, inference_parser_, data_reader_, reader_resource_manager,
        sparse_input_map_32_, reader_label_tensor_list_, reader_dense_tensor_list_, source,
        data_reader_type, check_type, slot_size_array, true);  // repeat dataset
  }
  tqdm bar;
  for (size_t batch = 0; batch < num_batches; batch++) {
    current_batch_size_ = data_reader_->read_a_batch_to_device();
    HCTR_CHECK_HINT(current_batch_size_ == global_max_batch_size_,
                    "there should not be imcomplete batch under the repeat mode");
    if (inference_params_.i64_input_key) {
      parse_input_from_data_reader<long long>(sparse_input_map_64_, key_tensor_list_64_,
                                              rowoffset_tensor_list_);
    } else {
      parse_input_from_data_reader<unsigned int>(sparse_input_map_32_, key_tensor_list_32_,
                                                 rowoffset_tensor_list_);
    }
#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
    {
      size_t i = omp_get_thread_num();
      CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
      long long current_batchsize_per_device =
          current_batch_size_ / resource_manager_->get_local_gpu_count();
      if (inference_params_.i64_input_key) {
        inference_sessions_[i]->predict(
            dense_tensor_list_[i].get_ptr(), key_tensor_list_64_[i].get_ptr(),
            rowoffset_tensor_list_[i].get_ptr(), pred_tensor_list_[i].get_ptr(),
            current_batchsize_per_device);
      } else {
        inference_sessions_[i]->predict(
            dense_tensor_list_[i].get_ptr(), key_tensor_list_32_[i].get_ptr(),
            rowoffset_tensor_list_[i].get_ptr(), pred_tensor_list_[i].get_ptr(),
            current_batchsize_per_device);
      }
      size_t pred_output_offset = (batch * current_batch_size_ + i * current_batchsize_per_device) *
                                  inference_parser_.label_dim;
      HCTR_LIB_THROW(cudaMemcpyAsync(
          pred_output + pred_output_offset, pred_tensor_list_[i].get_ptr(),
          current_batchsize_per_device * inference_parser_.label_dim * sizeof(float),
          cudaMemcpyDeviceToHost, resource_manager_->get_local_gpu(i)->get_stream()));
    }
    bar.progress(batch, num_batches);
  }
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
    HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(i)->get_stream()));
  }
  bar.finish();
}

float InferenceModel::evaluate(const size_t num_batches, const std::string& source,
                               const DataReaderType_t data_reader_type, const Check_t check_type,
                               const std::vector<long long>& slot_size_array) {
  size_t batch_size_per_gpu = global_max_batch_size_ / resource_manager_->get_local_gpu_count();
  metric_.reset(new metrics::AUC<float>(batch_size_per_gpu, num_batches, resource_manager_));
  for (size_t i = 0; i < resource_manager_->get_local_gpu_count(); i++) {
    raw_metrics_map_list_.push_back({{metrics::RawType::Pred, pred_tensor_list_[i].shrink()},
                                     {metrics::RawType::Label, label_tensor_list_[i].shrink()}});
  }

  reset_reader_tensor_list();
  // TODO: currently data reader can only be created on a single GPU to read the file list that may
  // contain only one parquet/binary file
  std::shared_ptr<ResourceManager> reader_resource_manager{
      ResourceManagerCore::create({{inference_params_.deployed_devices[0]}}, 0)};
  if (inference_params_.i64_input_key) {
    create_datareader<long long>()(
        inference_params_, inference_parser_, data_reader_, reader_resource_manager,
        sparse_input_map_64_, reader_label_tensor_list_, reader_dense_tensor_list_, source,
        data_reader_type, check_type, slot_size_array, true);  // repeat dataset
  } else {
    create_datareader<unsigned int>()(
        inference_params_, inference_parser_, data_reader_, reader_resource_manager,
        sparse_input_map_32_, reader_label_tensor_list_, reader_dense_tensor_list_, source,
        data_reader_type, check_type, slot_size_array, true);  // repeat dataset
  }
  tqdm bar;
  for (size_t batch = 0; batch < num_batches; batch++) {
    current_batch_size_ = data_reader_->read_a_batch_to_device();
    HCTR_CHECK_HINT(current_batch_size_ == global_max_batch_size_,
                    "there should not be imcomplete batch under the repeat mode");
    metric_->set_current_batch_size(current_batch_size_);
    if (inference_params_.i64_input_key) {
      parse_input_from_data_reader<long long>(sparse_input_map_64_, key_tensor_list_64_,
                                              rowoffset_tensor_list_);
    } else {
      parse_input_from_data_reader<unsigned int>(sparse_input_map_32_, key_tensor_list_32_,
                                                 rowoffset_tensor_list_);
    }
#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
    {
      size_t i = omp_get_thread_num();
      CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
      long long current_batchsize_per_device =
          current_batch_size_ / resource_manager_->get_local_gpu_count();
      if (inference_params_.i64_input_key) {
        inference_sessions_[i]->predict(
            dense_tensor_list_[i].get_ptr(), key_tensor_list_64_[i].get_ptr(),
            rowoffset_tensor_list_[i].get_ptr(), pred_tensor_list_[i].get_ptr(),
            current_batchsize_per_device);
      } else {
        inference_sessions_[i]->predict(
            dense_tensor_list_[i].get_ptr(), key_tensor_list_32_[i].get_ptr(),
            rowoffset_tensor_list_[i].get_ptr(), pred_tensor_list_[i].get_ptr(),
            current_batchsize_per_device);
      }
      metric_->local_reduce(i, raw_metrics_map_list_[i]);
    }
    metric_->global_reduce(resource_manager_->get_local_gpu_count());
    bar.progress(batch, num_batches);
  }
  float auc = metric_->finalize_metric();
  bar.finish();
  return auc;
}

template <typename TypeKey>
void InferenceModel::parse_input_from_data_reader(
    const std::map<std::string, SparseInput<TypeKey>>& sparse_input_map,
    std::vector<Tensor2<TypeKey>>& key_tensor_list,
    std::vector<Tensor2<int>>& rowoffset_tensor_list) {
#pragma omp parallel num_threads(resource_manager_->get_local_gpu_count())
  {
    size_t i = omp_get_thread_num();
    CudaDeviceContext context(resource_manager_->get_local_gpu(i)->get_device_id());
    size_t current_batch_size_per_gpu =
        current_batch_size_ / resource_manager_->get_local_gpu_count();

    std::vector<TypeKey> h_reader_value(current_batch_size_per_gpu *
                                        inference_parser_.max_feature_num_per_sample);
    std::vector<std::vector<TypeKey>> h_reader_rowoffset_list;
    size_t value_stride = 0;
    size_t rowoffset_stride = 0;
    for (size_t j = 0; j < inference_parser_.num_embedding_tables; j++) {
      size_t rowoffset_start =
          i * current_batch_size_per_gpu * inference_parser_.slot_num_for_tables[j];
      size_t rowoffset_length =
          current_batch_size_per_gpu * inference_parser_.slot_num_for_tables[j] + 1;
      SparseInput<TypeKey> sparse_input;
      if (!find_item_in_map(sparse_input, inference_parser_.sparse_names[j], sparse_input_map)) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Cannot find " + inference_parser_.sparse_names[j]);
      }
      // TODO: currently data reader can only be created on a single GPU to read the file list that
      // may contain only one parquet/binary file
      Tensor2<TypeKey> value_tensor = sparse_input.evaluate_sparse_tensors[0].get_value_tensor();
      Tensor2<TypeKey> rowoffset_tensor =
          sparse_input.evaluate_sparse_tensors[0].get_rowoffset_tensor();

      std::vector<TypeKey> h_reader_rowoffset(rowoffset_length);
      std::vector<int> h_reader_rowoffset_int(rowoffset_length);
      HCTR_LIB_THROW(cudaMemcpyAsync(h_reader_rowoffset.data(),
                                     rowoffset_tensor.get_ptr() + rowoffset_start,
                                     rowoffset_length * sizeof(TypeKey), cudaMemcpyDeviceToHost,
                                     resource_manager_->get_local_gpu(i)->get_stream()));
      HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(i)->get_stream()));
      size_t num_keys = h_reader_rowoffset.back() - h_reader_rowoffset.front();
      HCTR_LIB_THROW(cudaMemcpyAsync(h_reader_value.data() + value_stride,
                                     value_tensor.get_ptr() + h_reader_rowoffset.front(),
                                     num_keys * sizeof(TypeKey), cudaMemcpyDeviceToHost,
                                     resource_manager_->get_local_gpu(i)->get_stream()));

      TypeKey tmp = h_reader_rowoffset.front();
      for (auto& entry : h_reader_rowoffset) {
        entry -= tmp;
      }
      h_reader_rowoffset_list.push_back(h_reader_rowoffset);
      std::transform(h_reader_rowoffset.begin(), h_reader_rowoffset.end(),
                     h_reader_rowoffset_int.begin(), [](int x) { return static_cast<int>(x); });
      HCTR_LIB_THROW(cudaMemcpyAsync(rowoffset_tensor_list_[i].get_ptr() + rowoffset_stride,
                                     h_reader_rowoffset_int.data(), rowoffset_length * sizeof(int),
                                     cudaMemcpyHostToDevice,
                                     resource_manager_->get_local_gpu(i)->get_stream()));
      value_stride += num_keys;
      rowoffset_stride += rowoffset_length;
    }
    // TODO: currently data reader can only be created on a single GPU to read the file list that
    // may contain only one parquet/binary file
    Tensor2<float> dense_tensor = Tensor2<float>::stretch_from(reader_dense_tensor_list_[0]);
    Tensor2<float> label_tensor = Tensor2<float>::stretch_from(reader_label_tensor_list_[0]);
    HCTR_LIB_THROW(cudaMemcpyAsync(
        dense_tensor_list_[i].get_ptr(),
        dense_tensor.get_ptr() + i * current_batch_size_per_gpu * inference_parser_.dense_dim,
        current_batch_size_per_gpu * inference_parser_.dense_dim * sizeof(float),
        cudaMemcpyDeviceToDevice, resource_manager_->get_local_gpu(i)->get_stream()));
    HCTR_LIB_THROW(cudaMemcpyAsync(
        label_tensor_list_[i].get_ptr(),
        label_tensor.get_ptr() + i * current_batch_size_per_gpu * inference_parser_.label_dim,
        current_batch_size_per_gpu * inference_parser_.label_dim * sizeof(float),
        cudaMemcpyDeviceToDevice, resource_manager_->get_local_gpu(i)->get_stream()));
    HCTR_LIB_THROW(cudaStreamSynchronize(resource_manager_->get_local_gpu(i)->get_stream()));
    distribute_keys_for_inference(key_tensor_list[i].get_ptr(), h_reader_value.data(),
                                  current_batch_size_per_gpu, h_reader_rowoffset_list,
                                  inference_parser_.slot_num_for_tables);
  }
}

}  // namespace HugeCTR