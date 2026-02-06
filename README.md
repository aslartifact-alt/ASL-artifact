# ASL Artifact

This repository contains the source code and benchmarking scripts accompanying the submission on **Adaptive Synchronization Locks (ASL)**. The artifact provides multiple lock implementations, benchmark programs, and automation scripts to evaluate their performance under various concurrency scenarios.

---

## Artifact Overview

The artifact focuses on **lock implementations**, contention behavior, and classic concurrency problems, emphasizing **empirical performance comparison**. All files are self-contained C++ source code and shell scripts, requiring only a standard C++ compiler.

---

## Contents

### Lock Implementations

- `mutex.cpp` — Baseline using standard mutex for mutual exclusion.  
- `ttas.cpp` — Test-and-Test-and-Set (TTAS) spinlock.  
- `mcs.cpp` — Mellor-Crummey and Scott (MCS) queue-based spinlock.  
- `asl.cpp` — Adaptive spinlock used for comparative evaluation.  
- `hem.cpp` — Hemlock: compact and scalable mutual exclusion.  
- `exponential.cpp` — Spinlock with exponential backoff to reduce contention.  

### Concurrency Benchmarks

- `pc.cpp` — Producer–Consumer benchmark for evaluating synchronization under coordinated workloads.  

### Benchmark Scripts

- `runMutexBench.sh` — Automates performance benchmarking of all lock implementations.  
- `runProducerConsumer.sh` — Executes the producer–consumer benchmark across multiple configurations.  

---

## Build and Execution

All C++ source files require **C++11 or later** (e.g., `g++` or `clang++`).  

### Steps to Reproduce

The scripts will:

Compile the relevant source files.

Run experiments across multiple thread counts and configurations.

Output performance statistics (duration, total operations, throughput).

Automatically save results to CSV files (e.g., MutexBenchmark.csv) recording lock type, thread count, run number, and measured throughput.

Run the desired benchmark:
```bash
chmod +x runMutexBench.sh runProducerConsumer.sh
./runMutexBench.sh
./runProducerConsumer.sh

