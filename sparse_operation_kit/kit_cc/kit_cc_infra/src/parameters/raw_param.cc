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

#include "parameters/raw_param.h"
#include "tensor_buffer/tensor2_wrapper.h"
#include "embeddings/embedding_layer.h"
#include "parameters/dumping_functions.h"
#include "common.h"
#include <system_error>
#include <fstream>

namespace SparseOperationKit {

RawParam::RawParam(const std::string& initializer, const std::vector<size_t> shape,
                   const std::shared_ptr<ResourcesManager>& resource_mgr,
                   const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
                   const std::string var_name, const bool trainable)
: resource_mgr_(resource_mgr), 
hashtables_(resource_mgr->get_local_gpu_count()),
max_vocabulary_size_per_gpu_(shape[0]), embedding_vector_size_(shape[1]),
var_name_(var_name), trainable_(trainable), initializer_(Initializer::Get(initializer)),
has_hashtable_(true)
{
    emb_table_tensors_.reserve(resource_mgr_->get_local_gpu_count());
    emb_table_tensors_interface_.reserve(resource_mgr_->get_local_gpu_count());

    HugeCTR::CudaDeviceContext device_context;
    for (size_t dev_id = 0; dev_id < resource_mgr->get_local_gpu_count(); ++dev_id) {
        device_context.set_device(resource_mgr_->get_local_gpu(dev_id)->get_local_device_id());
        // reserve spaces for embedding table
        {
            Tensor2<float> tensor;
            buffers[dev_id]->reserve(shape, &tensor);
            emb_table_tensors_.push_back(tensor);
            emb_table_tensors_interface_.push_back(Tensor2Wrapper<float>::create(tensor));
        }
        
        // construct hashtable
        {
            hashtables_[dev_id].reset(new NvHashTable(max_vocabulary_size_per_gpu_));
        }
    } // for dev_id

    if (emb_table_tensors_.size() != emb_table_tensors_interface_.size())
        throw std::runtime_error(ErrorBase + "The size of embedding table tensors and its interface if not equal.");
}

RawParam::~RawParam() {}

std::shared_ptr<RawParam> RawParam::create(const std::string& initializer, const bool use_hashtable,
                                            const std::vector<size_t> shape,
                                            const std::shared_ptr<ResourcesManager>& resource_mgr,
            const std::vector<std::shared_ptr<HugeCTR::GeneralBuffer2<HugeCTR::CudaAllocator>>>& buffers,
                                            const std::string var_name, const bool trainable) {
    if (use_hashtable)
        return std::shared_ptr<RawParam>(new RawParam(initializer, shape, resource_mgr, buffers, var_name, trainable));
    else 
        throw std::runtime_error(ErrorBase + "Not implemented yet.");
}

size_t RawParam::get_max_vocabulary_size_per_gpu() const {
    return max_vocabulary_size_per_gpu_;
}

size_t RawParam::get_embedding_vec_size() const {
    return embedding_vector_size_;
}

void RawParam::init(const size_t global_replica_id) {
    const size_t local_replica_id = resource_mgr_->cal_local_id_from_global_id(global_replica_id);
    MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + 
            std::to_string(global_replica_id) + " start initialization");
    if (local_replica_id >= emb_table_tensors_.size()) 
        throw std::runtime_error(ErrorBase + "local_replica_id is out of the range of emb_table_tensors.size().");

    const auto &local_gpu = resource_mgr_->get_local_gpu(local_replica_id);

    initializer_->fill(emb_table_tensors_interface_[local_replica_id],
                       local_gpu->get_sm_count(),
                       local_gpu->get_variant_curand_gen(),
                       local_gpu->get_stream());

    resource_mgr_->sync_gpu(local_replica_id);

    MESSAGE("Variable: " + var_name_ + " on global_replica_id: " + 
            std::to_string(global_replica_id) + " initialization done.");
}


bool RawParam::trainable() const {
    return trainable_;
}

void RawParam::set_user(std::shared_ptr<EmbeddingLayer>& embedding) {
    user_ = embedding;
}

auto RawParam::get_hashtable(const size_t local_replica_id) -> std::shared_ptr<NvHashTable>&  {
    if (has_hashtable_) return hashtables_[local_replica_id];
    else throw std::runtime_error(ErrorBase + "Hashtable is not valid.");
}

std::shared_ptr<Tensor>& RawParam::get_embedding_table_tensor(const size_t local_replica_id) {
    if (local_replica_id >= emb_table_tensors_.size())
        throw std::runtime_error(ErrorBase + "local_replica_id is out of the range of emb_table_tensors.size().");

    return emb_table_tensors_interface_[local_replica_id];
}

std::string RawParam::get_var_name() const {
    return var_name_;
}

void RawParam::dump_to_file(const std::string filepath) {
    const size_t local_gpu_count = resource_mgr_->get_local_gpu_count();
    const size_t worker_id = resource_mgr_->get_worker_id();
    const size_t worker_num = resource_mgr_->get_workers_num();

    // step 1: get the count of key-index pairs on each hashtable on local worker.
    std::unique_ptr<size_t []> count;
    size_t local_worker_max_count = 0;
    if (0 == worker_id) { // chief worker
        count.reset(new size_t[worker_num * local_gpu_count]());
    } else {
        count.reset(new size_t[local_gpu_count]());
    }

    HugeCTR::CudaDeviceContext device_context;
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        device_context.set_device(local_gpu->get_local_device_id());

        const size_t hashtable_size = hashtables_[dev_id]->get_size(local_gpu->get_stream());
        if (hashtable_size != hashtables_[dev_id]->get_value_head(local_gpu->get_stream()))
            throw std::runtime_error(ErrorBase + " hashtable get_value_head() not equal to get_size().");
        if (hashtable_size > max_vocabulary_size_per_gpu_)
            throw std::runtime_error(ErrorBase + " keys count on GPU: " + std::to_string(dev_id) +
                                     " is out of the range of max_vocabulary_size_per_gpu.");

        count[dev_id] = hashtable_size;
        MESSAGE("Worker: " + std::to_string(worker_id) + ", GPU: " + std::to_string(dev_id) + 
                " key-index count = " + std::to_string(hashtable_size));
        local_worker_max_count = (local_worker_max_count > count[dev_id]) ? local_worker_max_count : count[dev_id];
    } // for dev_id in local_gpu_count

    // step 2: gather count among all workers
    size_t *d_global_max_count = nullptr;
    std::vector<size_t *> d_count(local_gpu_count);
    size_t *d_count_aggregation = nullptr;
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        device_context.set_device(local_gpu->get_local_device_id());

        if (0 == worker_id && 0 == dev_id) { // chief worker and chief GPU
            CK_CUDA(cudaMalloc(&d_global_max_count, sizeof(size_t) * 1));
            CK_CUDA(cudaMalloc(&d_count_aggregation, sizeof(size_t) * worker_num * local_gpu_count));
        }
        CK_CUDA(cudaMalloc(&d_count[dev_id], sizeof(size_t) * 1));
        CK_CUDA(cudaMemcpyAsync(d_count[dev_id], &count[dev_id], sizeof(size_t) * 1,
                                cudaMemcpyHostToDevice,
                                local_gpu->get_stream()));
    } // for dev_id in local_gpu_count

    resource_mgr_->sync_all_workers();

    CK_NCCL(ncclGroupStart());
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        CK_NCCL(ncclReduce(d_count[dev_id], d_global_max_count, 1, ncclUint64, ncclMax,
                           /*root=*/0, local_gpu->get_nccl(), local_gpu->get_stream()));
    } // for dev_id in local_gpu_count
    CK_NCCL(ncclGroupEnd());

    CK_NCCL(ncclGroupStart());
    if (0 == worker_id) { // chief worker
        const auto &local_gpu = resource_mgr_->get_local_gpu(0);
        device_context.set_device(local_gpu->get_local_device_id());
        for (size_t rank = 0; rank < resource_mgr_->get_global_gpu_count(); rank++) {
            CK_NCCL(ncclRecv(d_count_aggregation + rank, 1, ncclUint64, /*peer=*/rank,
                             local_gpu->get_nccl(),
                             local_gpu->get_stream()));
        } // for rank in global_gpu_count
    }
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        CK_NCCL(ncclSend(d_count[dev_id], 1, ncclUint64, /*peer=*/0,
                         local_gpu->get_nccl(),
                         local_gpu->get_stream()));
    } // for dev_id in local_gpu_count
    CK_NCCL(ncclGroupEnd());

    if (0 == worker_id) { // chief worker
        const auto &local_gpu = resource_mgr_->get_local_gpu(0);
        device_context.set_device(local_gpu->get_local_device_id());

        local_worker_max_count = 0;
        CK_CUDA(cudaMemcpyAsync(&local_worker_max_count, d_global_max_count, sizeof(size_t) * 1,
                                cudaMemcpyDeviceToHost,
                                local_gpu->get_stream()));
        CK_CUDA(cudaMemcpyAsync(count.get(), d_count_aggregation, 
                                sizeof(size_t) * worker_num * local_gpu_count,
                                cudaMemcpyDeviceToHost,
                                local_gpu->get_stream()));
        CK_CUDA(cudaStreamSynchronize(local_gpu->get_stream()));
    }

    // step 3: allocate temp spaces for dump parameters from GPU to CPU
    std::unique_ptr<int64_t *[]> h_hash_table_key(new int64_t *[local_gpu_count]);
    std::unique_ptr<int64_t *[]> d_hash_table_key(new int64_t *[local_gpu_count]);
    std::unique_ptr<int64_t *[]> d_hash_table_key_sort(new int64_t *[local_gpu_count]);
    std::unique_ptr<size_t *[]> h_hash_table_value_index(new size_t *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_hash_table_value_index(new size_t *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_hash_table_value_index_sort(new size_t *[local_gpu_count]);
    std::unique_ptr<float *[]> h_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<float *[]> d_hash_table_value(new float *[local_gpu_count]);
    std::unique_ptr<size_t *[]> d_dump_counter(new size_t *[local_gpu_count]);
    std::unique_ptr<void *[]> d_temp_storage(new void *[local_gpu_count]);
    std::vector<size_t> d_temp_storage_size(local_gpu_count, 0);
    for (size_t dev_id = 0; dev_id < local_gpu_count; ++dev_id) {
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        device_context.set_device(local_gpu->get_local_device_id());

        // TODO: use count[dev_id] instead of local_worker_max_count??
        CK_CUDA(cudaMallocHost(&h_hash_table_key[dev_id], local_worker_max_count * sizeof(int64_t))); 
        CK_CUDA(cudaMalloc(&d_hash_table_key[dev_id], local_worker_max_count * sizeof(int64_t)));
        CK_CUDA(cudaMalloc(&d_hash_table_key_sort[dev_id], local_worker_max_count * sizeof(int64_t)));
        CK_CUDA(cudaMallocHost(&h_hash_table_value_index[dev_id], local_worker_max_count * sizeof(size_t)));
        CK_CUDA(cudaMalloc(&d_hash_table_value_index[dev_id], local_worker_max_count * sizeof(size_t)));
        CK_CUDA(cudaMalloc(&d_hash_table_value_index_sort[dev_id], local_worker_max_count * sizeof(size_t)));
        CK_CUDA(cudaMallocHost(&h_hash_table_value[dev_id], local_worker_max_count * embedding_vector_size_ * sizeof(float)));
        CK_CUDA(cudaMalloc(&d_hash_table_value[dev_id], local_worker_max_count * embedding_vector_size_ * sizeof(float)));
        CK_CUDA(cudaMalloc(&d_dump_counter[dev_id], 1 * sizeof(size_t))); // FIXME: ???

        CK_CUDA(SortPairs((void*)nullptr, d_temp_storage_size[dev_id], 
                          /*d_keys_in=*/(size_t*)nullptr, /*d_keys_out=*/(size_t*)nullptr,
                          /*d_values_in=*/(int64_t*)nullptr, /*d_values_out=*/(int64_t*)nullptr,
                          /*num_items=*/local_worker_max_count));
        CK_CUDA(cudaMalloc(&d_temp_storage[dev_id], d_temp_storage_size[dev_id]));
    } // for dev_id in local_gpu_count
    resource_mgr_->sync_all_workers();

    // step 4: dump parameters to temp spaces
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        if (0 == count[dev_id]) continue;
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        device_context.set_device(local_gpu->get_local_device_id());
        MESSAGE("Worker: " + std::to_string(worker_id) + ", GPU: " + std::to_string(dev_id) + 
                ": dumping parameters from hashtable..");

        // get hashtable key-index pairs.
        hashtables_[dev_id]->dump(d_hash_table_key[dev_id], d_hash_table_value_index[dev_id],
                                  d_dump_counter[dev_id], local_gpu->get_stream());

        // sort key-index pairs.
        const int32_t end_bit = static_cast<int32_t>(std::log2(static_cast<float>(max_vocabulary_size_per_gpu_))) + 1;
        CK_CUDA(SortPairs(/*d_temp_storage=*/d_temp_storage[dev_id],
                          /*temp_storage_bytes=*/d_temp_storage_size[dev_id],
                          /*d_keys_in=*/d_hash_table_value_index[dev_id],
                          /*d_keys_out=*/d_hash_table_value_index_sort[dev_id],
                          /*d_values_in=*/d_hash_table_key[dev_id],
                          /*d_values_out=*/d_hash_table_key_sort[dev_id],
                          /*num_items=*/count[dev_id],
                          /*begin_bit=*/0,
                          /*end_bit*/end_bit,
                          local_gpu->get_stream(), false));

        // get embedding vector by sorted index
        get_hash_value(count[dev_id], embedding_vector_size_, d_hash_table_value_index_sort[dev_id],
                       emb_table_tensors_[dev_id].get_ptr(), d_hash_table_value[dev_id],
                       local_gpu->get_stream());
    } // for dev_id in local_gpu_count

    // step 5: save parameters to file stream. 
    // Only cheif worker needs to copy parameters from GPU to CPU, and then write it to file,
    // all the other workers only send their datas to cheif worker via NCCL.
    constexpr size_t key_size = sizeof(int64_t);
    constexpr size_t index_size = sizeof(size_t);
    const size_t values_size = sizeof(float) * embedding_vector_size_;
    std::unique_ptr<char []> key_buf(new char[local_worker_max_count * key_size]());
    std::unique_ptr<char []> index_buf(new char[local_worker_max_count * index_size]());
    std::unique_ptr<char []> embedding_value_buf(new char[local_worker_max_count * values_size]());

    if (0 == worker_id) { // on cheif worker
        const std::string key_filename = filepath + "/" + var_name_ + "_key.file";
        const std::string index_filename = filepath + "/" + var_name_ + "_index.file";
        const std::string values_filename = filepath + "/" + var_name_ + "_values.file";
        std::ofstream key_stream(key_filename, std::ios::binary | std::ios::out);
        std::ofstream index_stream(index_filename, std::ios::binary | std::ios::out);
        std::ofstream values_stream(values_filename, std::ios::binary | std::ios::out);

        size_t count_offset = 0;
        for (size_t worker = 0; worker < worker_num; worker++) {
            if (worker_id != worker) { /*cheif worker receives data from other workers*/ 
                CK_NCCL(ncclGroupStart());
                for (size_t recv_worker = 1; recv_worker < worker_num; recv_worker++) {
                    if (worker == recv_worker) { /*cheif worker receives valid data from other worker*/ 
                        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
                            const int32_t peer = worker * local_gpu_count + dev_id;
                            const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
                            const size_t pair_count = count[recv_worker * local_gpu_count + dev_id];
                            CK_NCCL(ncclRecv(d_hash_table_key_sort[dev_id], 
                                             pair_count,
                                             ncclInt64, /*peer=*/peer,
                                             local_gpu->get_nccl(),
                                             local_gpu->get_stream()));
                            CK_NCCL(ncclRecv(d_hash_table_value_index_sort[dev_id],
                                             pair_count,
                                             ncclUint64, /*peer=*/peer,
                                             local_gpu->get_nccl(),
                                             local_gpu->get_stream()));
                            CK_NCCL(ncclRecv(d_hash_table_value[dev_id], 
                                             pair_count * embedding_vector_size_,
                                             ncclFloat32, /*peer=*/peer,
                                             local_gpu->get_nccl(),
                                             local_gpu->get_stream()));
                        } // for dev_id in local_gpu_count
                        MESSAGE("Worker: " + std::to_string(worker) + "'s data is received by cheif node.");
                    } else { /*cheif worker receives dummy data from other worker*/
                        for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
                            CK_NCCL(ncclRecv(d_count[dev_id], 1, ncclUint64,
                                             /*peer=*/recv_worker * local_gpu_count + dev_id,
                                             resource_mgr_->get_local_gpu(dev_id)->get_nccl(),
                                             resource_mgr_->get_local_gpu(dev_id)->get_stream()));
                        } // for dev_id in local_gpu_count
                    }
                } // for recv_worker in [1, worker_num)
                CK_NCCL(ncclGroupEnd());
            } 

            /*cheif worker copy data from GPU to CPU, and save it to file stream.*/
            for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
                const size_t pair_count = count[worker * local_gpu_count + dev_id];
                const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
                CK_CUDA(cudaMemcpyAsync(h_hash_table_key[dev_id], d_hash_table_key_sort[dev_id],
                                        pair_count * sizeof(int64_t),
                                        cudaMemcpyDeviceToHost,
                                        local_gpu->get_stream()));

                indexes_add_offset(d_hash_table_value_index_sort[dev_id], pair_count, count_offset);

                CK_CUDA(cudaMemcpyAsync(h_hash_table_value_index[dev_id], d_hash_table_value_index_sort[dev_id],
                                        pair_count * sizeof(size_t),
                                        cudaMemcpyDeviceToHost,
                                        local_gpu->get_stream()));
                CK_CUDA(cudaMemcpyAsync(h_hash_table_value[dev_id], d_hash_table_value[dev_id],
                                        pair_count * embedding_vector_size_ * sizeof(float),
                                        cudaMemcpyDeviceToHost,
                                        local_gpu->get_stream()));
                count_offset += pair_count;
            } // for dev_id in local_gpu_count
            resource_mgr_->sync_local_gpus();

            /*save to file stream*/
            for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
                const size_t pair_count = count[worker * local_gpu_count + dev_id];
                // save sorted keys
                std::memcpy(key_buf.get(), h_hash_table_key[dev_id], pair_count * key_size);
                key_stream.write(key_buf.get(), pair_count * key_size);
                // save sorted indexes
                std::memcpy(index_buf.get(), h_hash_table_value_index[dev_id], pair_count * index_size);
                index_stream.write(index_buf.get(), pair_count * index_size);
                // save embedding vectors
                std::memcpy(embedding_value_buf.get(), h_hash_table_value[dev_id], pair_count * values_size);
                values_stream.write(embedding_value_buf.get(), pair_count * values_size);
                MESSAGE("Worker: " + std::to_string(worker) + ", GPU: " + std::to_string(dev_id) + 
                        "'s parameters saved to file.");
            } // for dev_id in local_gpu_count

        } // for worker in worker_num

        key_stream.close();
        index_stream.close();
        values_stream.close();
    } else { // non-cheif worker
        for (size_t worker = 1; worker < worker_num; worker++) {
            if (worker == worker_id) { /*sub worker send valid data to cheif worker*/
                CK_NCCL(ncclGroupStart());
                for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
                    const size_t pair_count = count[dev_id];
                    const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
                    const int32_t peer = 0 * local_gpu_count + dev_id;
                    CK_NCCL(ncclSend(d_hash_table_key_sort[dev_id], 
                                     pair_count, ncclInt64, 
                                     /*peer=*/peer, 
                                     local_gpu->get_nccl(), local_gpu->get_stream()));
                    CK_NCCL(ncclSend(d_hash_table_value_index_sort[dev_id], 
                                     pair_count, ncclUint64,
                                     /*peer=*/peer, 
                                     local_gpu->get_nccl(), local_gpu->get_stream()));
                    CK_NCCL(ncclSend(d_hash_table_value[dev_id], 
                                     pair_count * embedding_vector_size_,
                                     ncclFloat32, /*peer=*/peer,
                                     local_gpu->get_nccl(), local_gpu->get_stream()));
                } // for dev_id in local_gpu_count
                CK_NCCL(ncclGroupEnd()); 
                MESSAGE("Worker: " + std::to_string(worker) + "'s data sent to cheif worker.");
            } else { /*sub worker send dummy data to cheif worker*/
                CK_NCCL(ncclGroupStart());
                for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
                    CK_NCCL(ncclSend(d_count[dev_id], 1, ncclUint64, 
                                     /*peer=*/0 * local_gpu_count + dev_id,
                                     resource_mgr_->get_local_gpu(dev_id)->get_nccl(),
                                     resource_mgr_->get_local_gpu(dev_id)->get_stream()));
                }
                CK_NCCL(ncclGroupEnd());
            }
        } // for worker in [1, worker_num)
    }

    // step 6: synchronize all workers
    resource_mgr_->sync_all_workers();

    // finnaly: release temp spaces.
    for (size_t dev_id = 0; dev_id < local_gpu_count; dev_id++) {
        const auto &local_gpu = resource_mgr_->get_local_gpu(dev_id);
        device_context.set_device(local_gpu->get_local_device_id());

        if (0 == worker_id && 0 == dev_id) {
            CK_CUDA(cudaFree(d_global_max_count));
            CK_CUDA(cudaFree(d_count_aggregation));
        }
        CK_CUDA(cudaFree(d_count[dev_id]));
        CK_CUDA(cudaFreeHost(h_hash_table_key[dev_id]));
        CK_CUDA(cudaFree(d_hash_table_key[dev_id]));
        CK_CUDA(cudaFree(d_hash_table_key_sort[dev_id]));
        CK_CUDA(cudaFreeHost(h_hash_table_value_index[dev_id]));
        CK_CUDA(cudaFree(d_hash_table_value_index[dev_id]));
        CK_CUDA(cudaFree(d_hash_table_value_index_sort[dev_id]));
        CK_CUDA(cudaFreeHost(h_hash_table_value[dev_id]));
        CK_CUDA(cudaFree(d_hash_table_value[dev_id]));
        CK_CUDA(cudaFree(d_dump_counter[dev_id]));
        CK_CUDA(cudaFree(d_temp_storage[dev_id]));
    } // for dev_id in local_gpu_count
}

void RawParam::let_user_dump_to_file(const std::string filepath) {
    user_->dump_to_file(filepath);
}

void RawParam::restore_from_file(const std::string filename) {
    MESSAGE("Restoring " + var_name_ + " from " + filename);

    /* for RawParam, the content need to be restored from file
    *  is related to the details of embedding, so delegate
    *  this job to embedding layer.*/
    try {
        /*all nodes reads the file simultaneously.*/
        std::ifstream param_stream(filename, std::ifstream::binary);
        user_->restore_from_file(param_stream);
        param_stream.close();
    } catch (const std::system_error& error) {
        throw std::runtime_error(ErrorBase + error.what());
    }

    MESSAGE("Restored.");
}

void RawParam::load_tensors_to_memory(const std::vector<std::shared_ptr<Tensor>>& tensors) {
    /*for RawParam, how to load tensors to GPU memory 
    * is related to the details of embedding, so delegate
    * this job to embedding layer*/
    MESSAGE("Loading tensors to GPU memory.");

    // FIXME: when this function is called. the internal states need to be reset.

    user_->load_tensors_to_memory(tensors);

    MESSAGE("Loaded.");
}

} // namespace SparseOperationKit