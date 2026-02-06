#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <algorithm>

// --------------------------
// Mutex Lock (pthreads)
// --------------------------
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// Global variables
std::atomic<bool> stop_flag(false);
std::atomic<uint64_t> total_ops(0);

// Simulated critical section
int CS(int id) {
    return id * 2 + 1; // Dummy computation
}

// Worker thread function
void* Worker(void* arg) {
    int id = (intptr_t)arg;

    while (!stop_flag.load(std::memory_order_relaxed)) {
        pthread_mutex_lock(&mutex);
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);
        pthread_mutex_unlock(&mutex);
    }

    return nullptr;
}

int main(int argc, char* argv[]) {
    int num_threads = std::thread::hardware_concurrency();
    int test_duration_sec = 5;

    if (argc > 1) num_threads = std::stoi(argv[1]);
    if (argc > 2) test_duration_sec = std::stoi(argv[2]);

    std::vector<pthread_t> threads(num_threads);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Start worker threads
    for (int i = 0; i < num_threads; ++i)
        pthread_create(&threads[i], nullptr, Worker, (void*)(intptr_t)i);

    std::this_thread::sleep_for(std::chrono::seconds(test_duration_sec));
    stop_flag.store(true);

    // Wait for all threads
    for (int i = 0; i < num_threads; ++i)
        pthread_join(threads[i], nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end_time - start_time).count();
    uint64_t ops = total_ops.load();
    double mops = ops / (duration * 1e6);

    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Duration: " << duration << " sec\n";
    std::cout << "Total Ops: " << ops << "\n";
    std::cout << "Throughput: " << mops << " MOPS\n";

    pthread_mutex_destroy(&mutex);
    return 0;
}
