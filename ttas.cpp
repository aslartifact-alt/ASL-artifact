#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <pthread.h>
#include <algorithm>

// --------------------------
// Test-Test-and-Set Lock
// --------------------------
class TTASLock {
private:
    std::atomic<bool> locked;

public:
    TTASLock() : locked(false) {}

    void lock() {
        while (true) {
            // First test (read-only)
            while (locked.load(std::memory_order_relaxed)) {
                asm volatile("pause" ::: "memory");
            }
            // Try to acquire (write)
            if (!locked.exchange(true, std::memory_order_acquire))
                break;
        }
    }

    void unlock() {
        locked.store(false, std::memory_order_release);
    }
};

// Global TTAS lock
TTASLock lock;

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
        lock.lock();
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);
        lock.unlock();
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

    return 0;
}
