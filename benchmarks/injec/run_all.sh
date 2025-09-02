#!/bin/bash
set -euo pipefail

benchmarks=("directharm" "datastealing")

run_benchmark() {
    local abs_model="$1"
    local conc_model="$2"
    local abs_temp="$3"
    local conc_temp="$4"
    for benchmark in "${benchmarks[@]}"; do
        echo "Running: abs_model=$abs_model, conc_model=$conc_model, benchmark=$benchmark"
        uv run python -m benchmarks.injec.run \
            --abs_model "$abs_model" \
            --conc_model "$conc_model" \
            --abs_temperature "$abs_temp" \
            --conc_temperature "$conc_temp" \
            --embedding_threshold -1.0 \
            --benchmark_name "$benchmark" \
            --extra_context
    done
}

# GPT-4o
# run_benchmark "gpt-4o-2024-11-20" "gpt-4o-2024-11-20" 0.0 0.0
# run_benchmark "gpt-4o-2024-11-20" "gpt-4o-mini-2024-07-18" 0.0 0.0
# run_benchmark "gpt-4o-2024-11-20" "gpt-4.1-mini-2025-04-14" 0.0 0.0
# run_benchmark "gpt-4o-2024-11-20" "gpt-4.1-nano-2025-04-14" 0.0 0.0

# GPT-4.1
# run_benchmark "gpt-4.1-2025-04-14" "gpt-4.1-2025-04-14" 0.0 0.0
# run_benchmark "gpt-4.1-2025-04-14" "gpt-4o-mini-2024-07-18" 0.0 0.0
run_benchmark "gpt-4.1-2025-04-14" "gpt-4.1-mini-2025-04-14" 0.0 0.0
# run_benchmark "gpt-4.1-2025-04-14" "gpt-4.1-nano-2025-04-14" 0.0 0.0