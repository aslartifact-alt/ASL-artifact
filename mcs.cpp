#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <pthread.h>
#include <algorithm>

#if defined(__GNUC__) || defined(__clang__)
#define CALIGN alignas(64)
#else
#define CALIGN
#endif

// --------------------------
// MCS Lock
// --------------------------
class MCSLock {
public:
    struct Node { 
        std::atomic<Node*> next{nullptr}; 
        std::atomic<bool> locked{false}; 
    };
private:
    CALIGN std::atomic<Node*> tail{nullptr};
public:
    void lock(Node* node) {
        node->next.store(nullptr, std::memory_order_relaxed);
        node->locked.store(true, std::memory_order_relaxed);
        Node* pred = tail.exchange(node, std::memory_order_acq_rel);
        if (pred) {
            pred->next.store(node, std::memory_order_release);
            while (node->locked.load(std::memory_order_acquire)) {
#if defined(__i386__) || defined(__x86_64__)
                asm volatile("pause" ::: "memory");
#endif
            }
        }
    }
    void unlock(Node* node) {
        Node* succ = node->next.load(std::memory_order_acquire);
        if (!succ) {
            Node* expected = node;
            if (tail.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel)) return;
            while ((succ = node->next.load(std::memory_order_acquire)) == nullptr);
        }
        succ->locked.store(false, std::memory_order_release);
    }
};

// --------------------------
// Global MCSLock
// --------------------------
MCSLock lock;

// Global control variables
std::atomic<bool> stop_flag(false);
std::atomic<uint64_t> total_ops(0);

// Simulated critical section
int CS(int id) {
    return id * 2 + 1; // Dummy computation
}

// Worker thread function
void* Worker(void* arg) {
    intptr_t id = (intptr_t)arg;
    CALIGN MCSLock::Node myNode; // Each thread gets its own node

    while (!stop_flag.load(std::memory_order_relaxed)) {
        lock.lock(&myNode);
        CS(id);
        total_ops.fetch_add(1, std::memory_order_relaxed);
        lock.unlock(&myNode);
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
