#include <vector>
#include <cstdlib>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <omp.h>

#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <profiler.hpp>
#include <common.hpp>
#include <nlohmann/json.hpp>

#if CUDA_VERSION < 11010
#define CUDA_EVENT_RECORD_GRAPH(...) cudaEventRecord(__VA_ARGS__)
#else
#define CUDA_EVENT_RECORD_GRAPH(...) cudaEventRecordWithFlags(__VA_ARGS__, cudaEventRecordExternal)
#endif

using nlohmann::json;

namespace HugeCTR {

  Profiler::GPUTimer::GPUTimer() {
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&iter_start_, cudaEventBlockingSync));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&start_, cudaEventBlockingSync));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&stop_, cudaEventBlockingSync));
  }

  Profiler::GPUTimer::~GPUTimer() {
    cudaEventDestroy(iter_start_);
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void Profiler::GPUTimer::iter_start(cudaStream_t stream, bool use_cuda_graph) {
    if (use_cuda_graph) {
      cudaError_t retval = CUDA_EVENT_RECORD_GRAPH(iter_start_, stream);
      if (retval != cudaSuccess) {
        // some layers are not in cuda stream captured mode. so fall back to the normal cudaEventRecord
        CK_CUDA_THROW_(cudaEventRecord(iter_start_, stream));
      }
    } else {
      CK_CUDA_THROW_(cudaEventRecord(iter_start_, stream));
    }
  }

  void Profiler::GPUTimer::event_start(cudaStream_t stream, bool use_cuda_graph) {
    if (use_cuda_graph) {
      cudaError_t retval = CUDA_EVENT_RECORD_GRAPH(start_, stream);
      if (retval != cudaSuccess) {
        // some layers are not in cuda stream captured mode. so fall back to the normal cudaEventRecord
        CK_CUDA_THROW_(cudaEventRecord(start_, stream));
      }
    } else {
      CK_CUDA_THROW_(cudaEventRecord(start_, stream));
    }
  }

  void Profiler::GPUTimer::event_stop(cudaStream_t stream, bool use_cuda_graph) {
    if (use_cuda_graph) {
      cudaError_t retval = CUDA_EVENT_RECORD_GRAPH(stop_, stream);
      if (retval != cudaSuccess) {
        // some layers are not in cuda stream captured mode. so fall back to the normal cudaEventRecord
        CK_CUDA_THROW_(cudaEventRecord(stop_, stream));
      }
    } else {
      CK_CUDA_THROW_(cudaEventRecord(stop_, stream));
    }
  }

  void Profiler::GPUTimer::sync_stop() {
    CK_CUDA_THROW_(cudaEventSynchronize(stop_));
  }

  float Profiler::GPUTimer::get_iter_start_to_event_start_ms() {
    float iter_start_to_event_start_ms;
    CK_CUDA_THROW_(cudaEventElapsedTime(&iter_start_to_event_start_ms, iter_start_, start_));
    return iter_start_to_event_start_ms;
  }

  float Profiler::GPUTimer::get_measured_time_ms() {
    float measured_time_ms;
    CK_CUDA_THROW_(cudaEventElapsedTime(&measured_time_ms, start_, stop_));
    return measured_time_ms;
  }

  void Profiler::initialize(bool use_cuda_graph) {
    char* pd = std::getenv("PROFILING_DIR");
    if (pd == NULL) {
      std::string msg("Got empty for env PROFILING_DIR. You must specify if when using this profiler");
      ERROR_MESSAGE_(msg);
      throw std::invalid_argument(msg);
    }
    profiling_dir_ = std::string(pd);
    MESSAGE_(std::string("Profiler using PROFILING_DIR: ") + profiling_dir_);

    interested_events_ = {};
    std::ifstream interested_events_file((profiling_dir_ + "/" + "prof.events").c_str());
    if (interested_events_file.good()) {
      MESSAGE_(std::string("Profiler using prof.events inside PROFILING_DIR"));
      for (std::string line; getline(interested_events_file, line);) {
        interested_events_.push_back(line);
      }
    }

    char* warmup_iterations_str = std::getenv("PROFILING_WARMUP_ITERS");
    if (warmup_iterations_str == NULL) {
      warmup_iterations_ = 10;
    } else {
      warmup_iterations_ = std::atoi(warmup_iterations_str) + 1;
    }
    MESSAGE_(std::string("Profiler using WARMUP_ITERS: ") + std::to_string(warmup_iterations_ - 1));

    char* repeat_times_str = std::getenv("PROFILING_REPEAT_TIMES_PER_EVENT");
    if (repeat_times_str == NULL) {
      repeat_times_ = 50;
    } else {
      repeat_times_ = std::atoi(repeat_times_str);
    }
    MESSAGE_(std::string("Profiler using REPEAT_TIMES_PER_EVENT: ") + std::to_string(repeat_times_));

    char* warmup_after_cudagraph_reinit_str = std::getenv("PROFILING_WARMUP_AFTER_CUDAGRAPH_REINIT");
    if (warmup_after_cudagraph_reinit_str == NULL) {
      warmup_after_cudagraph_reinit_ = 10;
    } else {
      warmup_after_cudagraph_reinit_ = std::atoi(warmup_after_cudagraph_reinit_str);
    }
    MESSAGE_(std::string("Profiler using PROFILING_WARMUP_AFTER_CUDAGRAPH_REINIT: ") + std::to_string(warmup_after_cudagraph_reinit_));

    // for extra cuda graph init iter, it won't count
    repeat_times_ += warmup_after_cudagraph_reinit_ + 1;
    current_reapted_times_ = 0;

    char host_name[50];
    gethostname(host_name, 50);
    host_name_ = std::string(host_name);
    use_cuda_graph_ = use_cuda_graph;
    if (CUDA_VERSION < 11010 && use_cuda_graph_) {
      MESSAGE_(std::string("CUDA version is ") + std::to_string(CUDA_VERSION) +
               ". Profiler do not support use_cuda_graph = true and CUDA version < 11.1 at the same time." +
               " Consider use higher CUDA version or use_cuda_graph = false. Program exit.");
      std::exit(0);
    }

    MESSAGE_(std::string("Profiler using cuda graph: ") + std::to_string(use_cuda_graph_));
    current_iteration_ = 0;
    current_event_idx_ = 0;
    init_cuda_graph_this_iter = false;
  }


  void Profiler::iter_check() {
    auto end = std::chrono::steady_clock::now();
    for(auto& x : map_internal_) {
      for(auto it = x.second->begin(); it != x.second->end(); it++) {
        it->second = 0;
      }
    }
    if (current_iteration_ <= warmup_iterations_ - 1) {
        current_iteration_ += 1;
        return;
    } else {
      if (events_.size() == 0) {
        MESSAGE_("No profiling labels found int code or they are not in prof.events. Please have a check. Program exit.");
        std::exit(0);
      }
    }
    if (current_iteration_ == warmup_iterations_) {
      // first iteration
      current_iteration_ += 1;
      prepare_iter_start();
    } else {
      // not first iteration
      // whether to record result logic block
      if(!init_cuda_graph_this_iter) {
        if (!use_cuda_graph_ || (use_cuda_graph_ && current_reapted_times_ > warmup_after_cudagraph_reinit_)) {
          iter_time_ms_.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - iter_check_).count() / 1000000.0
          );
          auto gpu_timer = map_stream_to_gpu_timer_[static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream];
          events_[current_event_idx_]->measured_times_ms.push_back(gpu_timer->get_measured_time_ms());
          events_[current_event_idx_]->iter_start_to_event_start_times_ms.push_back(gpu_timer->get_iter_start_to_event_start_ms());
        }
      }
      current_iteration_ += 1;
      current_reapted_times_ += 1;

      if (current_reapted_times_ >= repeat_times_) {
        current_reapted_times_ = 0;
        current_event_idx_ += 1;
      }
      if (current_event_idx_ >= int(events_.size())) {
          // finish iterate events, should write result and end
          MESSAGE_(std::string("Profiling complete! Result json file is wrote under PROFILING_DIR. Program exit."));
          write_result();
          std::exit(0);
      }
      prepare_iter_start();
    }
    // MESSAGE_(std::string("Iter: ") + std::to_string(current_iteration_) + " event_idx: " +
    //          std::to_string(current_event_idx_) + " repeat_times: " +  std::to_string(current_reapted_times_) +
    //          " init cudagraph: " + std::to_string(init_cuda_graph_this_iter));
  }

  void Profiler::prepare_iter_start() {
    if (use_cuda_graph_) {
      if(current_reapted_times_ == 0) {
        MESSAGE_(std::string("Iteration: ") + std::to_string(current_iteration_) +
                 std::string(". Profiler re-instantiate cuda graph for ") +
                 std::to_string(current_event_idx_) + " : " +
                 events_[current_event_idx_]->event_name + " " +
                 std::to_string(
                   static_cast<GPUEvent*>(events_[current_event_idx_].get())->met_times_within_this_stream
                 ) + " on " +
                 stream_str(static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream)
        );
        init_cuda_graph_this_iter = true;
      } else {
        init_cuda_graph_this_iter = false;
      }
    } else {
      if(current_reapted_times_ == 0) {
        MESSAGE_(std::string("Iteration: ") + std::to_string(current_iteration_) +
                 ". " + std::to_string(current_event_idx_) + " : " +
                 events_[current_event_idx_]->event_name + " " +
                std::to_string(
                  static_cast<GPUEvent*>(events_[current_event_idx_].get())->met_times_within_this_stream
                ) + " on " +
                stream_str(static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream)
        );
      }
      init_cuda_graph_this_iter = false;
    }

    for(auto& s_and_gt : map_stream_to_gpu_timer_) {
      s_and_gt.second->iter_start(s_and_gt.first, false);
    }
    iter_check_ = std::chrono::steady_clock::now();
  }

  void Profiler::record_event(const char* event_label_char, cudaStream_t stream, int device_id) {
    try {
      // auto t_start = std::chrono::steady_clock::now();
      std::string event_label = std::string(event_label_char);
      int dot_pos = event_label.find_last_of(std::string("."));
      std::string event_type = event_label.substr(dot_pos + 1);
      if (event_type != "start" && event_type != "stop") {
        throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
        std::string("Invalid event name. Should end with .start or .stop"));
      }
      std::string event_name = event_label.substr(0, dot_pos);
      // above string operation cost 0.000xxx ms on DGXA100. x usually is 1 - 2.
      if (current_iteration_ <= warmup_iterations_) {
        mtx_.lock();
        thread_local int current_device_id;
        cudaGetDevice(&current_device_id);
        if (current_device_id != device_id) {
          CK_CUDA_THROW_(cudaSetDevice(device_id));
        }

        auto map_iter = map_stream_to_gpu_timer_.find(stream);
        if(map_iter == map_stream_to_gpu_timer_.end()) {
          auto gpu_timer = std::make_shared<GPUTimer>();
          map_stream_to_gpu_timer_[stream] = gpu_timer;
          map_internal_[stream] = std::make_shared<std::map<std::string, int>>();
        }
        int met_times_within_this_stream;
        try {
          met_times_within_this_stream = map_internal_[stream]->at(event_name);
        } catch (const std::out_of_range& e) {
          map_internal_[stream]->insert({event_name, 0});
          met_times_within_this_stream = 0;
        }

        if (interested_events_.size() > 0) {
          if (std::find(interested_events_.begin(), interested_events_.end(), event_name) == interested_events_.end()) {
            if (event_type == "stop") {
              map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
            }
            if (current_device_id != device_id) {
              CK_CUDA_THROW_(cudaSetDevice(current_device_id));
            }
            mtx_.unlock();
            return;
          }
        }
        std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
        int event_idx = find_event(event_key);

        if (event_type == "start") {
          if(event_idx >= 0) {
            // event exist!
            if (current_device_id != device_id) {
              CK_CUDA_THROW_(cudaSetDevice(current_device_id));
            }
            mtx_.unlock();
            return;
          }

          // create new event
          auto gpu_event = new GPUEvent;
          gpu_event->event_name = event_name;
          gpu_event->met_times_within_this_stream = met_times_within_this_stream;
          gpu_event->start_index = events_num_;
          gpu_event->end_index = -1; // wait for stop event to set,
          gpu_event->measured_times_ms = std::vector<float>();
          gpu_event->iter_start_to_event_start_times_ms = std::vector<float>();
          gpu_event->device_id = device_id;
          gpu_event->stream = stream;
          events_.push_back(std::shared_ptr<Event>(static_cast<Event*>(gpu_event)));
          map_event_key_to_event_idx[event_key] = events_.size() - 1;
          events_num_++;
          //PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " + std::to_string(met_times_within_this_stream));
          if (current_device_id != device_id) {
            CK_CUDA_THROW_(cudaSetDevice(current_device_id));
          }
        }
        else {
          // event_name == "stop"
          // only update the end_index
          if (event_idx >= 0) {
            auto event = events_[event_idx];
            if (event->end_index < 0) {
              event->end_index = events_num_;
              events_num_++;
            }
            map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
            //PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " + std::to_string(met_times_within_this_stream));
          } else {
            throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
              std::string("Event ") + event_name + std::string(" has stop but no start"));
          }
        }
        if (current_device_id != device_id) {
          CK_CUDA_THROW_(cudaSetDevice(current_device_id));
        }
        mtx_.unlock();
      }
      else {
        int met_times_within_this_stream = map_internal_[stream]->at(event_name);
        if (events_[current_event_idx_]->event_name != event_name ||
            static_cast<GPUEvent*>(events_[current_event_idx_].get())->stream != stream ||
            static_cast<GPUEvent*>(events_[current_event_idx_].get())->met_times_within_this_stream != met_times_within_this_stream) {
              if (event_type == "stop") {
                map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
              }
              return;
        }
        // above map and if compare costs 0.000x ms on DGXA100, x is usually 1 - 7.
        thread_local int current_device_id;
        cudaGetDevice(&current_device_id);
        if (current_device_id != device_id) {
          CK_CUDA_THROW_(cudaSetDevice(device_id));
        }
        auto gpu_timer = map_stream_to_gpu_timer_[stream];
        // above getdevice and mapping costs 0.000x ms on DGXA100, x is usually 1 - 2.

        if (event_type == "start") {
          gpu_timer->event_start(stream, use_cuda_graph_);
        } else {
          gpu_timer->event_stop(stream, use_cuda_graph_);
          // event_start and event_stop usually costs 0.002ms on DGXA100
          map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
          // Above post event record operation costs 0.00x on DGXA100, usually x is 1 - 2.
        }
        if (current_device_id != device_id) {
          CK_CUDA_THROW_(cudaSetDevice(current_device_id));
        }
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  int Profiler::find_event(std::string& event_key) {
    int idx = -1;
    try {
      idx = map_event_key_to_event_idx.at(event_key);
    } catch (const std::out_of_range& e) {}
    return idx;
  }

  void Profiler::write_result() {
    int ret = std::system((std::string("mkdir -p ") + profiling_dir_).c_str());
    if (ret != 0) {
      MESSAGE_("Creating PROFILING_DIR failed?");
    }
    std::string result_file = profiling_dir_ + '/' + host_name_ + ".prof.json";

    json result;
    result["host_name"] = host_name_;
    result["iter_time_ms"] = iter_time_ms_;
    result["events"] = json::array();
    for (auto& event_p : events_) {
      GPUEvent* gep = static_cast<GPUEvent*>(event_p.get());
      json j;
      j["event_name"] = gep->event_name;
      j["device_id"] = gep->device_id;
      j["stream"] = stream_str(gep->stream);
      j["start_index"] = gep->start_index;
      j["end_index"] = gep->end_index;
      j["met_times_within_this_stream"] = gep->met_times_within_this_stream;
      j["measured_times_ms"] = gep->measured_times_ms;
      j["iter_start_to_event_start_times_ms"] = gep->iter_start_to_event_start_times_ms;
      result["events"].push_back(j);
    }
    std::string result_jstring = result.dump();
    std::ofstream outfile;
    outfile.open(result_file.c_str());
    outfile << result_jstring;
    outfile.close();
  }

  // A global variable
  Profiler global_profiler;

}  // namespace HugeCTR
