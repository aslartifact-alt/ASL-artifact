#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <random>
#include <chrono>
#include <cmath>
#include <pthread.h>
#include <algorithm>
#include <fstream>

// RLSpinLock and QLearningAgent Implementation
enum Action { SPIN = 0, YIELD = 1, SLEEP = 2 };
const int NUM_ACTIONS = 3;
const int NUM_STATES = 10;
const char* QTABLE_FILENAME = "qtable.dat";

void initializeWarmQ(std::vector<std::vector<double>>& Q) {
    for (int state = 0; state < NUM_STATES; ++state) {
        for (int action = 0; action < NUM_ACTIONS; ++action) {
            if (state < 3) {
                Q[state][SPIN] = 1.0;
                Q[state][YIELD] = 0.5;
                Q[state][SLEEP] = 0.1;
            } else if (state < 7) {
                Q[state][SPIN] = 0.2;
                Q[state][YIELD] = 1.0;
                Q[state][SLEEP] = 0.3;
            } else {
                Q[state][SPIN] = 0.1;
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

        // Try to load Q-table from file
        if (!loadQTable(QTABLE_FILENAME)) {
            // If loading failed, initialize warm Q-table
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
        int maxIdx = 0;
        double maxVal = row[0];
        for (int i = 1; i < NUM_ACTIONS; ++i) {
            if (row[i] > maxVal) {
                maxVal = row[i];
                maxIdx = i;
            }
        }
        return static_cast<Action>(maxIdx);
    }

    void update(int state, Action action, double reward, int nextState) {
        double maxNextQ = *std::max_element(Q[nextState].begin(), Q[nextState].end());
        Q[state][action] += alpha * (reward + gamma * maxNextQ - Q[state][action]);
    }

    void decayEpsilon() {
        epsilon = std::max(0.01, epsilon * 0.99995);
    }

    void saveQTable(const char* filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs) {
            std::cerr << "Failed to open Q-table file for writing.\n";
            return;
        }
        for (int s = 0; s < NUM_STATES; ++s) {
            ofs.write(reinterpret_cast<const char*>(Q[s].data()), NUM_ACTIONS * sizeof(double));
        }
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
        // Fast path: try immediate acquisition
        if (!locked.exchange(true, std::memory_order_acquire)) {
            // Acquired lock immediately - fast path success
            return;
        }

        // Slow path: RL-guided adaptive locking on failure of fast path
        int failed_attempts = 0;
        struct Stats { int state; Action action; };
        std::vector<Stats> attempts_stats;

        while (true) {
            int state = std::min(NUM_STATES - 1, (int)std::log2(failed_attempts + 1));
            Action action = agent.selectAction(state);
            attempts_stats.push_back({state, action});

            switch (action) {
                case SPIN:
                    for (int i = 0; i < 50 + 10 * state; ++i)
                        asm volatile("pause" ::: "memory");
                    break;
                case YIELD:
                    std::this_thread::yield();
                    break;
                case SLEEP:
                    std::this_thread::sleep_for(std::chrono::microseconds(10 + 5 * state));
                    break;
            }

            if (!locked.exchange(true, std::memory_order_acquire))
                break;

            failed_attempts++;
        }

        double reward = -static_cast<double>(failed_attempts);
        for (auto& stat : attempts_stats) {
            if (++update_count % 10 == 0)
                agent.update(stat.state, stat.action, reward, 0);
        }

        agent.decayEpsilon();
    }

    void unlock() {
        locked.store(false, std::memory_order_release);
    }

    void saveQTable() {
        agent.saveQTable(QTABLE_FILENAME);
    }

private:
    alignas(64) std::atomic<bool> locked;
    QLearningAgent agent;
    size_t update_count;
};

// Global variables
RLSpinLock lock;
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

    // Save Q-table on exit for offline use
    lock.saveQTable();

    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Duration: " << duration << " sec\n";
    std::cout << "Total Ops: " << ops << "\n";
    std::cout << "Throughput: " << mops << " MOPS\n";

    return 0;
}
