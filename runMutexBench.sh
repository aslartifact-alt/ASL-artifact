#!/bin/bash

# -------------------------
# List of lock source files
# -------------------------
SOURCES=("exponential.cpp" "mcs.cpp" "hem.cpp" "ttas.cpp" "asl.cpp" "mutex.cpp")

# -------------------------
# Output CSV file
# -------------------------
OUTPUT="MutexBenchmark.csv"

# -------------------------
# Benchmark parameters
# -------------------------
DURATION=60          # Seconds per run
CS_ITERS=10000       # Work inside critical section (if used by source code)
THREAD_COUNTS="1 2 4 8"
REPS=10              # Repetitions per thread count

# -------------------------
# Prepare CSV header
# -------------------------
echo "LockType,Threads,Run,Throughput_MOPS" > $OUTPUT

# -------------------------
# Compile all programs
# -------------------------
echo "Compiling locks with -O3 optimization..."
for src in "${SOURCES[@]}"; do
    exe="${src%.cpp}"
    echo "Compiling $src -> $exe"
    g++ -O3 -std=c++17 -pthread "$src" -o "$exe"
    if [ $? -ne 0 ]; then
        echo "Error: Compilation failed for $src"
        exit 1
    fi
done

# -------------------------
# Run benchmarks
# -------------------------
for src in "${SOURCES[@]}"; do
    exe="${src%.cpp}"
    lock_name="$exe"
    echo "Benchmarking $lock_name..."

    for t in $THREAD_COUNTS; do
        for run in $(seq 1 $REPS); do
            echo "Running $lock_name with $t threads, run #$run..."
            OUTPUT_TEXT=$("./$exe" $t $DURATION)

            # Parse throughput from program output
            THROUGHPUT=$(echo "$OUTPUT_TEXT" | grep "Throughput:" | awk '{print $2}')

            # Append results to CSV
            echo "$lock_name,$t,$run,$THROUGHPUT" >> $OUTPUT
        done
    done
done

echo "All benchmarks completed. Results saved to $OUTPUT"
