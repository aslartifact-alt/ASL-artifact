/*
 g++ -O3 -Wall --std=c++20 benchmark.cpp -o benchmark.out -pthread

 Unified Benchmark for Embedded / Edge Devices
 ------------------------------------------------
 Lock types supported (select via command-line):
   mutex   - std::mutex
   ttas    - Test-and-Test-and-Set spinlock
   hem     - HEM Lock (replaces atomic)
   expbo   - Exponential Backoff Lock
   mcs     - MCS Lock
   asl     - RL-based Adaptive Spin Lock (ASL)

 Usage:
   ./benchmark.out <lock_type> <producers> <consumers> <iterations> <work_iterations>

 Example:
   ./benchmark.out hem 2 2 5000000 100
*/

#include <thread>
#include <deque>
#include <iostream>
#include <vector>
#include <chrono>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <random>
#include <fstream>
#include <cmath>
#include <cstdint>

// =====================================================================================
// HEM Lock
// =====================================================================================
using Grant = std::atomic<bool>;

class HemLock {
    alignas(64) std::atomic<Grant*> lock_ptr{nullptr};

public:
    void lock(Grant* myGrant) {
        myGrant->store(false, std::memory_order_relaxed);
        Grant* prev = lock_ptr.exchange(myGrant, std::memory_order_acq_rel);
        if (prev != nullptr) {
            while (!prev->exchange(false, std::memory_order_acq_rel)) {
#if defined(__i386__) || defined(__x86_64__)
                asm volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
                __asm__ __volatile__("yield");
#endif
            }
        }
    }

    void unlock(Grant* myGrant) {
        myGrant->store(true, std::memory_order_release);
        Grant* expected = myGrant;
        if (!lock_ptr.compare_exchange_strong(expected, nullptr, std::memory_order_release, std::memory_order_acquire)) {
            while (myGrant->load(std::memory_order_acquire)) {
#if defined(__i386__) || defined(__x86_64__)
                asm volatile("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
                __asm__ __volatile__("yield");
#endif
            }
        }
    }
};

class HEMWrapper {
public:
    HEMWrapper() {}
    void lock() { l.lock(&myGrant); }
    void unlock() { l.unlock(&myGrant); }
private:
    HemLock l;
    thread_local static Grant myGrant;
};
thread_local Grant HEMWrapper::myGrant{true};

// =====================================================================================
// ASL Lock (RLSpinLock + QLearningAgent)
// =====================================================================================
enum Action { SPIN = 0, YIELD = 1, SLEEP = 2 };
const int NUM_ACTIONS = 3;
const int NUM_STATES = 10;
const char* QTABLE_FILENAME = "qtable.dat";

void initializeWarmQ(std::vector<std::vector<double>>& Q) {
    for (int state = 0; state < NUM_STATES; ++state) {
        for (int action = 0; action < NUM_ACTIONS; ++action) {
            if (state < 3) {
                Q[state][SPIN]  = 1.0;
                Q[state][YIELD] = 0.5;
                Q[state][SLEEP] = 0.1;
            } else if (state < 7) {
                Q[state][SPIN]  = 0.2;
                Q[state][YIELD] = 1.0;
                Q[state][SLEEP] = 0.3;
            } else {
                Q[state][SPIN]  = 0.1;
                Q[state][YIELD] = 0.3;
                Q[state][SLEEP] = 1.0;
            }
        }
    }
}

class QLearningAgent {
public:
    QLearningAgent(double alpha = 0.05, double gamma = 0.95, double epsilon = 0.1)
        : alpha(alpha), gamma(gamma), epsilon(epsilon) {
        Q.resize(NUM_STATES, std::vector<double>(NUM_ACTIONS, 0.0));
        rng.seed(std::random_device{}());

        if (!loadQTable(QTABLE_FILENAME)) {
            initializeWarmQ(Q);
            std::cout << "Initialized warm Q-table.\n";
        } else {
            std::cout << "Loaded Q-table from file.\n";
        }
    }

    Action selectAction(int state) {
        double p = dist(rng);
        if (p < epsilon) return static_cast<Action>(fastRand() % NUM_ACTIONS);
        auto& row = Q[state];
        int maxIdx = 0; double maxVal = row[0];
        for (int i = 1; i < NUM_ACTIONS; ++i) {
            if (row[i] > maxVal) { maxVal = row[i]; maxIdx = i; }
        }
        return static_cast<Action>(maxIdx);
    }

    void update(int state, Action action, double reward, int nextState) {
        double maxNextQ = *std::max_element(Q[nextState].begin(), Q[nextState].end());
        Q[state][action] += alpha * (reward + gamma * maxNextQ - Q[state][action]);
    }

    void decayEpsilon() { epsilon = std::max(0.01, epsilon * 0.99995); }

    void saveQTable(const char* filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) { std::cerr << "Failed to open Q-table file for writing.\n"; return; }
        for (int s = 0; s < NUM_STATES; ++s)
            ofs.write(reinterpret_cast<const char*>(Q[s].data()), NUM_ACTIONS * sizeof(double));
        ofs.close();
        std::cout << "Q-table saved to file.\n";
    }

private:
    bool loadQTable(const char* filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return false;
        for (int s = 0; s < NUM_STATES; ++s) {
            ifs.read(reinterpret_cast<char*>(Q[s].data()), NUM_ACTIONS * sizeof(double));
            if (!ifs) return false;
        }
        ifs.close();
        return true;
    }

    std::vector<std::vector<double>> Q;
    double alpha, gamma, epsilon;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist{0.0, 1.0};
    uint32_t xorshift_state = 123456789;

    uint32_t fastRand() {
        uint32_t x = xorshift_state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        xorshift_state = x;
        return x;
    }
};

class RLSpinLock {
public:
    RLSpinLock() : locked(false), update_count(0) {}

    void lock() {
        if (!locked.exchange(true, std::memory_order_acquire)) return;

        int failed_attempts = 0;
        struct Stats { int state; Action action; };
        std::vector<Stats> attempts_stats;

        while (true) {
            int state = std::min(NUM_STATES - 1, (int)std::log2(failed_attempts + 1));
            Action action = agent.selectAction(state);
            attempts_stats.push_back({state, action});

            switch (action) {
                case SPIN:
                    for (int i = 0; i < 50 + 10 * state; ++i) {
#if defined(__aarch64__) || defined(__arm__)
                        __asm__ __volatile__("yield");
#else
                        __asm__ __volatile__("pause");
#endif
                    }
                    break;
                case YIELD:
                    std::this_thread::yield();
                    break;
                case SLEEP:
                    std::this_thread::sleep_for(std::chrono::microseconds(10 + 5 * state));
                    break;
            }

            if (!locked.exchange(true, std::memory_order_acquire)) break;
            failed_attempts++;
        }

        double reward = -static_cast<double>(failed_attempts);
        for (auto& stat : attempts_stats) {
            if (++update_count % 10 == 0)
                agent.update(stat.state, stat.action, reward, 0);
        }
        agent.decayEpsilon();
    }

    void unlock() { locked.store(false, std::memory_order_release); }
    void saveQTable() { agent.saveQTable(QTABLE_FILENAME); }

private:
    alignas(64) std::atomic<bool> locked;
    QLearningAgent agent;
    size_t update_count;
};

// =====================================================================================
// Exponential Backoff Lock
// =====================================================================================
class ExponentialBackoffLock {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
public:
    void lock() {
        int delay = 1;
        while (true) {
            while (flag.test(std::memory_order_relaxed)) {
                for (int i = 0; i < delay; ++i) {
#if defined(__aarch64__) || defined(__arm__)
                    __asm__ __volatile__("yield");
#else
                    __asm__ __volatile__("pause");
#endif
                }
                if (delay < 1 << 12) delay <<= 1;
            }
            if (!flag.test_and_set(std::memory_order_acquire)) return;
        }
    }
    void unlock() { flag.clear(std::memory_order_release); }
};

// =====================================================================================
// MCS Lock
// =====================================================================================
struct MCSNode { std::atomic<MCSNode*> next{nullptr}; std::atomic<bool> locked{false}; };
class MCSLock {
    std::atomic<MCSNode*> tail{nullptr};
public:
    void lock(MCSNode* node) {
        node->next.store(nullptr, std::memory_order_relaxed);
        MCSNode* prev = tail.exchange(node, std::memory_order_acq_rel);
        if (prev) {
            node->locked.store(true, std::memory_order_relaxed);
            prev->next.store(node, std::memory_order_release);
            while (node->locked.load(std::memory_order_acquire)) {
#if defined(__aarch64__) || defined(__arm__)
                __asm__ __volatile__("yield");
#else
                __asm__ __volatile__("pause");
#endif
            }
        }
    }
    void unlock(MCSNode* node) {
        MCSNode* succ = node->next.load(std::memory_order_acquire);
        if (!succ) {
            MCSNode* expected = node;
            if (tail.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel)) return;
            while (!(succ = node->next.load(std::memory_order_acquire))) {
#if defined(__aarch64__) || defined(__arm__)
                __asm__ __volatile__("yield");
#else
                __asm__ __volatile__("pause");
#endif
            }
        }
        succ->locked.store(false, std::memory_order_release);
        node->next.store(nullptr, std::memory_order_relaxed);
    }
};

// =====================================================================================
// Lock wrappers (polymorphic interface)
// =====================================================================================
class LockInterface { public: virtual void lock()=0; virtual void unlock()=0; virtual ~LockInterface(){} };

class MutexWrapper : public LockInterface { std::mutex m; public: void lock() override { m.lock(); } void unlock() override { m.unlock(); } };
class TTASLock { std::atomic_flag flag=ATOMIC_FLAG_INIT; public: void lock() { while(true){while(flag.test(std::memory_order_relaxed)){} if(!flag.test_and_set(std::memory_order_acquire)) return; } } void unlock(){flag.clear(std::memory_order_release);} };
class TTASWrapper : public LockInterface { TTASLock l; public: void lock() override { l.lock(); } void unlock() override { l.unlock(); } };
class ExpBackoffWrapper : public LockInterface { ExponentialBackoffLock l; public: void lock() override { l.lock(); } void unlock() override { l.unlock(); } };
class MCSWrapper : public LockInterface { MCSLock l; thread_local static MCSNode node; public: void lock() override { l.lock(&node); } void unlock() override { l.unlock(&node); } };
thread_local MCSNode MCSWrapper::node;
class ASLWrapper : public LockInterface { RLSpinLock l; public: void lock() override { l.lock(); } void unlock() override { l.unlock(); } };
class HEMLockWrapper : public LockInterface { HEMWrapper l; public: void lock() override { l.lock(); } void unlock() override { l.unlock(); } };

// =====================================================================================
// Benchmark state
// =====================================================================================
int iterations = 5'000'000;
int producer_thread_count = 2;
int consumer_thread_count = 2;
int work_iterations = 100;
std::deque<int> queue;
std::atomic<int> producer_count{0};
std::atomic<int> consumer_count{0};
LockInterface* g_lock = nullptr;

// =====================================================================================
// Producer/Consumer workload
// =====================================================================================
inline void do_work(int &val) { for(int i=0;i<work_iterations;i++) val=(val*13+7)%9973; }
void producer() { for(int i=0;i<iterations/producer_thread_count;i++){ g_lock->lock(); queue.push_back(++producer_count); g_lock->unlock(); } }
void consumer() { while(true){ int val=0; bool has_value=false; g_lock->lock(); if(!queue.empty()){val=queue.front();queue.pop_front();has_value=true;} else if(producer_count>=iterations){g_lock->unlock(); return;} g_lock->unlock(); if(has_value){do_work(val); consumer_count++;} } }

// =====================================================================================
// Main
// =====================================================================================
int main(int argc,char* argv[]){
    if(argc<6){ std::cerr<<"Usage: ./benchmark.out <lock_type> <producers> <consumers> <iterations> <work_iterations>\n"; return 1; }
    std::string lock_type=argv[1];
    producer_thread_count=std::atoi(argv[2]);
    consumer_thread_count=std::atoi(argv[3]);
    iterations=std::atoi(argv[4]);
    work_iterations=std::atoi(argv[5]);

    if(lock_type=="mutex") g_lock=new MutexWrapper();
    else if(lock_type=="ttas") g_lock=new TTASWrapper();
    else if(lock_type=="hem") g_lock=new HEMLockWrapper();
    else if(lock_type=="expbo") g_lock=new ExpBackoffWrapper();
    else if(lock_type=="mcs") g_lock=new MCSWrapper();
    else if(lock_type=="asl") g_lock=new ASLWrapper();
    else { std::cerr<<"Unknown lock type: "<<lock_type<<"\n"; return 1; }

    using clock=std::chrono::high_resolution_clock;
    auto start=clock::now();
    std::vector<std::thread> producers, consumers;
    for(int i=0;i<producer_thread_count;i++) producers.emplace_back(producer);
    for(int i=0;i<consumer_thread_count;i++) consumers.emplace_back(consumer);
    for(auto &t:producers) t.join();
    for(auto &t:consumers) t.join();
    auto end=clock::now();
    std::chrono::duration<double> elapsed=end-start;

    std::cout<<"--- Benchmark Results ---\n";
    std::cout<<"Lock type:  "<<lock_type<<"\n";
    std::cout<<"Producers:  "<<producer_thread_count<<"\n";
    std::cout<<"Consumers:  "<<consumer_thread_count<<"\n";
    std::cout<<"Iterations: "<<iterations<<"\n";
    std::cout<<"CPU work per item: "<<work_iterations<<"\n";
    std::cout<<"Produced:   "<<producer_count.load()<<"\n";
    std::cout<<"Consumed:   "<<consumer_count.load()<<"\n";
    std::cout<<"Elapsed time: "<<elapsed.count()<<" sec\n";
    std::cout<<"Throughput:  "<<(consumer_count.load()/elapsed.count())<<" items/sec\n";

    delete g_lock;
    return 0;
}

