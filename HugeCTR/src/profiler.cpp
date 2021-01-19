#include <vector>
#include <cstdlib>
#include <string>
#include <map>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <omp.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <profiler.hpp>
#include <common.hpp>
#include <nlohmann/json.hpp>
using nlohmann::json;

namespace HugeCTR {

  Profiler::GPUTimer::GPUTimer() {
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&start_, cudaEventBlockingSync));
    CK_CUDA_THROW_(cudaEventCreateWithFlags(&stop_, cudaEventBlockingSync));
  }

  Profiler::GPUTimer::~GPUTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void Profiler::GPUTimer::event_start(cudaStream_t stream) {
    CK_CUDA_THROW_(cudaEventRecord(start_, stream));
  }

  void Profiler::GPUTimer::event_stop(cudaStream_t stream) {
    CK_CUDA_THROW_(cudaEventRecord(stop_, stream));
    CK_CUDA_THROW_(cudaEventSynchronize(stop_));
    CK_CUDA_THROW_(cudaEventElapsedTime(&measured_time_ms_, start_, stop_));
    // MESSAGE_("Result " + std::to_string(measured_time_ms_));
  }

  float Profiler::GPUTimer::get_result() {
    return measured_time_ms_;
  }

  void Profiler::initialize(const char* schedule_file) {
    // read from a schedule file, schedule file format:
    //    warmup_iterations
    //    event_name_1, iteration1,
    //    event_name_2, iteration2
    //    ...

    // TBD how to get host_name in CPP ?

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
    map_internal_.clear();
    // MESSAGE_(std::string("Current iter: " + std::to_string(current_iteration_)));
  }

  void Profiler::iter_end() {
    if (current_iteration_ > warmup_iterations_) {
      current_schedule_idx_++;
    }

    if (current_schedule_idx_ >= int(scheduled_events_.size())) {
        auto result_file = write_result("prof.json");
        MESSAGE_(std::string("Profiling complete! Result file is writing to ") + result_file + ". Program exit.");
        std::exit(0);
    }
    current_iteration_++;
  }

  void Profiler::record_event(const char* event_label_char, cudaStream_t stream, int device_id) {
    try {
      // event_label is xxx.xxx.start or xxx.xxx.end, parse suffix out of it
      auto event_label = std::string(event_label_char);
      int dot_pos = event_label.find_last_of(std::string("."));
      std::string event_type = event_label.substr(dot_pos + 1);

      if (event_type != "start" && event_type != "stop") {
        throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
        std::string("Invalid event name. Should end with .start or .stop"));
      }
      std::string event_name = event_label.substr(0, dot_pos);
      if (current_iteration_ <= warmup_iterations_) {
        mtx_.lock();
        int met_times_within_this_stream = safe_access_map_internel(event_name, stream);
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
          if (event_type == "stop") { map_internal_[event_name][stream] = met_times_within_this_stream + 1; }
          mtx_.unlock();
          return;
        }

        auto map_iter = map_stream_to_gpu_timer_.find(stream);
        if(map_iter == map_stream_to_gpu_timer_.end()) {
          auto gpu_timer = std::make_shared<GPUTimer>();
          map_stream_to_gpu_timer_[stream] = gpu_timer;
        }
        std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
        int event_idx = find_event(event_key);

        if (event_type == "start") {
          if(event_idx >= 0) {
            // event exist!
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
          gpu_event->device_id = device_id;
          gpu_event->stream = stream;
          events_.push_back(std::shared_ptr<Event>(static_cast<Event*>(gpu_event)));
          map_event_key_to_event_idx[event_key] = events_.size() - 1;
          events_num_++;
          //PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " + std::to_string(met_times_within_this_stream));

        } else { // event_name == "stop"
          // only update the end_index
          if (event_idx >= 0) {
            auto event = events_[event_idx];
            if (event->end_index < 0) {
              event->end_index = events_num_;
              events_num_++;
            }
            map_internal_[event_name][stream] = met_times_within_this_stream + 1;
            //PROFILER_DEBUG_(std::string("Parsed a new GPU event ") + event_label + " occured_time " + std::to_string(met_times_within_this_stream));
          } else {
            throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
              std::string("Event ") + event_name + std::string(" has stop but no start"));
          }

        }
        mtx_.unlock();
      } else {
        mtx_.lock();
        int met_times_within_this_stream = safe_access_map_internel(event_name, stream);
        if (std::get<0>(scheduled_events_[current_schedule_idx_]) != event_name || \
            std::get<1>(scheduled_events_[current_schedule_idx_]) != current_iteration_ || \
            std::get<3>(scheduled_events_[current_schedule_idx_]) != met_times_within_this_stream) {
              if (event_type == "stop") {
                map_internal_[event_name][stream] = met_times_within_this_stream + 1;
              }
              mtx_.unlock();
              return;
        }
        mtx_.unlock();
        auto gpu_timer = map_stream_to_gpu_timer_[stream];
        if (event_type == "start") {
          gpu_timer->event_start(stream);
        } else {
          gpu_timer->event_stop(stream);
          std::string event_key = gen_event_key(event_name, stream, met_times_within_this_stream);
          int event_idx = find_event(event_key);
          if (event_idx < 0) {
            throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
              std::string("Current event ") + event_name + std::string(" not registered!"));
          }
          mtx_.lock();
          events_[event_idx]->measured_times_ms.push_back(gpu_timer->get_result());
          map_internal_[event_name][stream] = met_times_within_this_stream + 1;
          mtx_.unlock();
          // PROFILER_DEBUG_(std::string("Timing on ") + event_label);
        }
      }
    } catch (const std::runtime_error& rt_err) {
      std::cerr << rt_err.what() << std::endl;
      throw;
    }
  }

  int Profiler::safe_access_map_internel(std::string& event_name, cudaStream_t stream) {
    int times = 0;
    try {
      map_internal_.at(event_name);
    } catch (const std::out_of_range& e) {
      map_internal_[event_name] = {};
    }
    try {
      times = map_internal_[event_name].at(stream);
    } catch (const std::out_of_range& e) {
      map_internal_[event_name][stream] = times;
    }
    return times;
  }

  int Profiler::find_event(std::string& event_key) {
    int idx = -1;
    try {
      idx = map_event_key_to_event_idx.at(event_key);
    } catch (const std::out_of_range& e) {}
    return idx;
  }

  std::string Profiler::write_result(const char* result_file) {
    // TBD dump events_ to json file
    json result = json::array();
    for (auto& event_p : events_) {
      GPUEvent* gep = static_cast<GPUEvent*>(event_p.get());
      json j;
      j["layer_name"] = gep->layer_name;
      j["name"] = gep->name;
      j["device_id"] = gep->device_id;
      j["stream"] = stream_str(gep->stream);
      j["start_index"] = gep->start_index;
      j["end_index"] = gep->end_index;
      j["measured_times_ms"] = gep->measured_times_ms;
      result.push_back(j);
    }
    std::string result_jstring = result.dump();
    std::ofstream outfile;
    outfile.open(result_file);
    outfile << result_jstring;
    outfile.close();
    return result_file;
  }

  // A global variable
  Profiler global_profiler;

}  // namespace HugeCTR
