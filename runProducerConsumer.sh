#!/bin/bash

# ---------------------------------------
# Benchmark Compilation & Runner Script
# ---------------------------------------
# Compiles the benchmark and runs all lock types under multiple producer/consumer configs
# Each experiment is repeated 5 times
# Stores all output in a single file (overwrites old results)
# ---------------------------------------

# Benchmark source and binary
SRC="pc.cpp"
BIN="pc.out"

# Lock types (added "hem")
LOCKS=("mutex" "ttas" "mcs" "expbo" "asl" "hem")
ITERATIONS=5000000
WORK_ITERS=100
REPEATS=5

# Test scenarios: (name producers consumers)
TESTS=(
  "balanced 2 2"
  "full_load 4 4"
)

OUTFILE="benchmark_results.log"

# ---------------------------------------
# Compile the benchmark
# ---------------------------------------
echo "Compiling $SRC..."
g++ -O3 -Wall --std=c++20 "$SRC" -o "$BIN" -pthread
if [ $? -ne 0 ]; then
    echo "Compilation failed. Exiting."
    exit 1
fi
echo "Compilation successful."

# ---------------------------------------
# Start fresh output file
# ---------------------------------------
{
  echo "Unified Benchmark Results"
  echo "=========================="
  echo "Run date: $(date)"
  echo ""
} > "$OUTFILE"

# ---------------------------------------
# Run benchmarks
# ---------------------------------------
for test in "${TESTS[@]}"; do
  set -- $test
  NAME=$1
  PROD=$2
  CONS=$3

  {
    echo "==========================================="
    echo " Running test: $NAME (Producers=$PROD, Consumers=$CONS)"
    echo "==========================================="
  } >> "$OUTFILE"

  for lock in "${LOCKS[@]}"; do
    for ((r=1; r<=REPEATS; r++)); do
      {
        echo "--- Lock: $lock | Run $r ---"
        ./"$BIN" "$lock" $PROD $CONS $ITERATIONS $WORK_ITERS
        echo ""
      } >> "$OUTFILE"
    done
  done
done

echo "All benchmarks complete. Results saved in $OUTFILE"

