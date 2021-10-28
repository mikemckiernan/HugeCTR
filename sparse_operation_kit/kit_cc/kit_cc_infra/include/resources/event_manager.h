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

#ifndef RESOURCES_EVENT_MANAGER_H
#define RESOURCES_EVENT_MANAGER_H

#include "resources/event.h"
#include <queue>
#include <mutex>
#include <thread>

namespace SparseOperationKit {

class EventManager {
public:
    static std::unique_ptr<EventManager> create();
    ~EventManager();

    EventManager(EventManager&) = delete;
    EventManager& operator=(EventManager&) = delete;
    EventManager(EventManager&&) = delete;
    EventManager& operator=(EventManager&&) = delete;

    std::shared_ptr<Event> get_event();
    void sync_two_streams(cudaStream_t& root_stream, 
                          cudaStream_t& sub_stream);

protected:
    EventManager();

private:
    std::queue<std::shared_ptr<Event>> unused_events_;
    std::queue<std::shared_ptr<Event>> inuse_events_;
    bool should_stop_;
    std::mutex mu_;
    std::thread porter_thread_;

    void porter_function();
};

} // namespace SparseOperationKit

#endif // RESOURCES_EVENT_MANAGER_H