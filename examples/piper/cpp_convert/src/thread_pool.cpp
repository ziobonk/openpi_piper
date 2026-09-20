#include "thread_pool.h"
#include <iostream>

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
    for (size_t i = 0; i < num_threads; ++i) {
        _workers.emplace_back(&ThreadPool::_worker_loop, this);
    }
}

ThreadPool::~ThreadPool() {
    _stop = true;
    _cv.notify_all();
    for (auto& w : _workers) {
        if (w.joinable()) w.join();
    }
}

void ThreadPool::_worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock lock(_mutex);
            _cv.wait(lock, [this] { return _stop || !_tasks.empty(); });
            if (_stop && _tasks.empty()) return;
            task = std::move(_tasks.front());
            _tasks.pop();
        }
        task();
    }
}
