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

  void Profiler::GPUTimer::iter_start(cudaStream_t stream) {
    CK_CUDA_THROW_(cudaEventRecord(iter_start_, stream));
  }

  void Profiler::GPUTimer::event_start(cudaStream_t stream) {
    CK_CUDA_THROW_(cudaEventRecord(start_, stream));
  }

  void Profiler::GPUTimer::event_stop(cudaStream_t stream) {
    CK_CUDA_THROW_(cudaEventRecord(stop_, stream));
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

  void Profiler::initialize() {
    try {
      profiling_dir_ = std::string(std::getenv("PROFILING_DIR"));
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      std::cerr << "Got empty for env PROFILING_DIR. You must specify if when using this profiler" << std::endl;
      throw;
    }
    char host_name[50];
    gethostname(host_name, 50);
    host_name_ = std::string(host_name);
    std::string schedule_file = profiling_dir_ + "/prof.schedule";
    MESSAGE_(std::string("Profiler initializing using ") + schedule_file + " ...");
    std::ifstream schedule_f(schedule_file);
    int line_no = 0;
    for (std::string line; getline(schedule_f, line);) {
        if (line_no) {
          auto splited = split_string(line, ' ');
          scheduled_events_.push_back(std::make_tuple(splited[0], std::stoi(splited[1]), splited[2], std::stoi(splited[3])));
        } else {
          warmup_iterations_ = std::stoi(line);
        }
        line_no++;
    }
    current_iteration_ = 1;
    current_schedule_idx_ = 0;
  }

  void Profiler::iter_start() {
    for(auto& x : map_internal_) {
      for(auto it = x.second->begin(); it != x.second->end(); it++) {
        it->second = 0;
      }
    }
    if(current_iteration_ > warmup_iterations_) {
      for(auto& s_and_gt : map_stream_to_gpu_timer_) {
        s_and_gt.second->iter_start(s_and_gt.first);
        s_and_gt.second->event_idx_for_this_iter = -1;
      }
    }
    iter_start_check_ = std::chrono::steady_clock::now();
  }

  void Profiler::iter_end() {
    if (current_iteration_ > warmup_iterations_) {
      for(auto& s_and_gt : map_stream_to_gpu_timer_) {
        int event_idx = s_and_gt.second->event_idx_for_this_iter;
        if (event_idx < 0) {
          // no event is recorded on this stream
          break;
        }
        s_and_gt.second->sync_stop();
        events_[event_idx]->measured_times_ms.push_back(s_and_gt.second->get_measured_time_ms());
        events_[event_idx]->iter_start_to_event_start_times_ms.push_back(s_and_gt.second->get_iter_start_to_event_start_ms());
      }

      iter_end_check_ = std::chrono::steady_clock::now();
      iter_time_ms_.push_back(
        std::chrono::duration_cast<std::chrono::nanoseconds>(iter_end_check_- iter_start_check_).count() / 1000000.0
      );

      current_schedule_idx_++;
    }

    if (current_schedule_idx_ >= int(scheduled_events_.size())) {
        std::string result_file = profiling_dir_ + '/' + host_name_ + ".prof.json";
        MESSAGE_(std::string("Profiling complete! Result file is writing to ") + result_file + ". Program exit.");
        write_result(result_file.c_str());
        std::exit(0);
    }
    current_iteration_++;
  }

  void Profiler::record_event(const char* event_label_char, cudaStream_t stream, int device_id) {
    try {
      // event_label is xxx.xxx.start or xxx.xxx.end, parse suffix out of it
      // auto t_start = std::chrono::steady_clock::now();
      std::string event_label = std::string(event_label_char);
      int dot_pos = event_label.find_last_of(std::string("."));
      std::string event_type = event_label.substr(dot_pos + 1);
      if (event_type != "start" && event_type != "stop") {
        throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
        std::string("Invalid event name. Should end with .start or .stop"));
      }
      std::string event_name = event_label.substr(0, dot_pos);

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

        // parse the event label, register it and create resources.
        bool found = false;
        std::string layer_name;
        for (auto& it : scheduled_events_) {
          if (std::get<0>(it) == event_name && std::get<3>(it) == met_times_within_this_stream) {
            layer_name = std::get<2>(it);
            found = true;
            break;
          }
        }
        if (!found) {
          if (event_type == "stop") {
            map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
          }
          if (current_device_id != device_id) {
            CK_CUDA_THROW_(cudaSetDevice(current_device_id));
          }
          mtx_.unlock();
          return;
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
          gpu_event->name = event_name;
          gpu_event->layer_name = layer_name;
          gpu_event->same_name_events_occured_order_in_code = met_times_within_this_stream;
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
        } else { // event_name == "stop"
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
      } else {
        int met_times_within_this_stream = map_internal_[stream]->at(event_name);
        if (std::get<0>(scheduled_events_[current_schedule_idx_]) != event_name || \
            std::get<1>(scheduled_events_[current_schedule_idx_]) != current_iteration_ || \
            std::get<3>(scheduled_events_[current_schedule_idx_]) != met_times_within_this_stream) {
              if (event_type == "stop") {
                map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
              }
              return;
        }

        thread_local int current_device_id;
        cudaGetDevice(&current_device_id);
        if (current_device_id != device_id) {
          CK_CUDA_THROW_(cudaSetDevice(device_id));
        }
        auto gpu_timer = map_stream_to_gpu_timer_[stream];
        if (event_type == "start") {
          gpu_timer->event_start(stream);
        } else {
          // auto t_end = std::chrono::steady_clock::now();
          gpu_timer->event_stop(stream);
          std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
          int event_idx = find_event(event_key);
          if (event_idx < 0) {
            throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
              std::string("Current event ") + event_name + std::string(" not registered!"));
          }
          gpu_timer->event_idx_for_this_iter = event_idx;
          // mtx_.lock();
          map_internal_[stream]->operator[](event_name) = met_times_within_this_stream + 1;
          // After measuring, the cpu overhead is around 0.000x ms.
          // auto prior_cpu_overhead_ms = std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() / 1000000.0 );
          // PROFILER_DEBUG_(std::string("CPU prior overhead ") + prior_cpu_overhead_ms);
          // mtx_.unlock();
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

  void Profiler::write_result(const char* result_file) {
    // TBD dump events_ to json file
    json result;
    result["host_name"] = host_name_;
    result["iter_time_ms"] = iter_time_ms_;
    result["events"] = json::array();
    for (auto& event_p : events_) {
      GPUEvent* gep = static_cast<GPUEvent*>(event_p.get());
      json j;
      j["layer_name"] = gep->layer_name;
      j["name"] = gep->name;
      j["device_id"] = gep->device_id;
      j["stream"] = stream_str(gep->stream);
      j["start_index"] = gep->start_index;
      j["end_index"] = gep->end_index;
      j["same_name_events_occured_order_in_code"] = gep->same_name_events_occured_order_in_code;
      j["measured_times_ms"] = gep->measured_times_ms;
      j["iter_start_to_event_start_times_ms"] = gep->iter_start_to_event_start_times_ms;
      result["events"].push_back(j);
    }
    std::string result_jstring = result.dump();
    std::ofstream outfile;
    outfile.open(result_file);
    outfile << result_jstring;
    outfile.close();
  }

  // A global variable
  Profiler global_profiler;

}  // namespace HugeCTR
