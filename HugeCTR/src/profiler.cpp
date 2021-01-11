#include <inline_profiler.hpp>

namespace HugeCTR {

namespace InlineProfiler {

  GPUTimer::GPUTimer(cudaStream_t stream) {
    stream_ = stream;
    cudaEventCreate(this->start_);
    cudaEventCreate(this->stop_);
  }

  GPUTimer::~GPUTimer() {
    cudaEventDestroy(this->start_);
    cudaEventDestroy(this->stop_);
  }

  void GPUTimer::event_start() {
    cudaEventRecord(this->start, stream_);
  }

  void GPUTimer::event_end() {
    cudaEventRecord(this->end, stream_);
  }

  float GPUTimer::get_result() {
    float elapsed_time;
    cudaElapsedTime(this->start, this->stop);
    return elapsed_time;
  }


  InlineProfiler() {};

  void initialize(std::string schedule_file) {
    // read from a schedule file, schedule file format:
    //    warmup_iterations
    //    event_name_1, iteration1,
    //    event_name_2, iteration2
    //    ...

    // TBD how to get host_name in CPP ?

    std::ifstream schedule_f(schedule_file);
    int line_no = 0;
    for (std::string line; getline(schedule_f, line);) {
        if (line_no) {
          auto splited = Helpers::split_string(line, ' ');
          scheduled_events_.push_back(std::make_pair<std::string, unsigned int>(splited.first, int(splited.second)));
        } else {
          warmup_iterations_ = int(line);
        }
        line_no++;
    }
    current_iteration_ = 0;
    current_schedule_idx_ = 0;
  }

  void iter_start() {
    map_event_id_current_gpu_timer_.clear();
  }

  void iter_end() {
    if (current_iteration_ > warmup_iterations_) {
      // get result;
      for(auto& it : map_event_id_current_gpu_timer_) {
        float result = it->second.get_result();
        events_[it.first].measured_time = result;
      }
      current_schedule_idx_++;
    }
    current_iteration_++;
  }

  void record_event(std::string event_label, cudaStream_t stream) {
    // event_label is xxx.xxx.start or xxx.xxx.end, parse suffix out of it
    int dot_pos = event_label.find_last_of(std::string('.'));
    std::string event_type = event_label.substr(dot_pos + 1);
    if (event_type != "start" || event_type != "stop") {
      throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
      std::string("Invalid event name. Should end with .start or .stop"))
    }
    std::string event_name = event_label.substr(0, dot_pos);

    if (current_iteration_ <= warmup_iterations_) {
      // parse the event label, register it and create resources.
      mtx_.lock();

      auto it = scheduled_events_.begin();
      for(; it != scheduled_events_.end(); it++ {
        if (it->first == event_name) { break;}
      }
      if (it == scheduled_events_.end()) { return; }

      auto map_iter = map_stream_gpu_timer_.find(stream);
      unsigned int stream_id = map_stream_id_[stream];

      if(map_iter == map_stream_gpu_timer_.end()) {
        auto gpu_timer = make_shared<GPUTimer>(stream);
        map_stream_gpu_timer_[stream] = gpu_timer;
        map_iter = map_stream_gpu_timer_.end();
        stream_id = distance(map_stream_gpu_timer_.begin(), map_iter);
        map_stream_id_[stream] = stream_id;
      }

      // get device id from stream
      unsigned int device_id = get_device_id(stream)

      if (event_type == "start") {
        // create new event
        events_.push_back(std::shared_ptr<Event>(new Event{
          event_name,
          events_num_,
          0, // wait for stop event to set,
          device_id,
          stream_id
        }));
      } else { // event_name == "end"
        // only update the end_index
        events_(event_idx)->end_index = events_num_;
      }
      events_num_++;
      mtx_.unlock();
    } else {
      if (current_event_name_ != event_name || current_iteration_ != current_scheduled_iter_) { return; }
      auto gpu_timer = map_stream_gpu_timer_[stream];
      if (event_type == "start") {
        gpu_timer->event_start();
      } else {
        gpu_timer->event_stop();
        int event_idx = find_event(event_name, stream);
        if (event_idx < 0) {
          throw internal_runtime_error(HugeCTR::Error_t::UnspecificError, \
            std::string("Current event ") + event_name + std::string::(" not registered!"));
        }
        mtx_.lock();
        map_event_id_current_gpu_timer_[event_idx] = gpu_timer;
        mtx.unlock();
      }
    }
  }

  int find_event(std::string event_name, cudaStream_t stream) {
    for (int i = 0; i < events_; i++) {
      if (events_[i].name == event_name && ) {
        return i;
      }
    }
    return -1;
  }

  void write_result(std::string result_file) {
    // TBD dump events_ to json file
  }

}  // namespace InlineProfiler

}  // namespace sHugeCTR

// A global variable global_inline_profiler;
HugeCTR::InlineProfiler global_inline_profiler();