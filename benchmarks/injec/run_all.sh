#!/bin/bash
set -euo pipefail

benchmarks=("directharm" "datastealing")

run_benchmark() {
    local abs_model="$1"
    local conc_model="$2"
    local abs_temp="$3"
    local conc_temp="$4"

    for benchmark in "${benchmarks[@]}"; do
        if [ -f "benchmarks/injec/results/${abs_model}-${conc_model}/${benchmark}/results.jsonl" ]; then
            echo "Results for abs_model=$abs_model, conc_model=$conc_model, benchmark=$benchmark already exist. Skipping..."
            continue
        fi

        echo "Running: abs_model=$abs_model, conc_model=$conc_model, benchmark=$benchmark"
        uv run python -m benchmarks.injec.run \
            --abs_model "$abs_model" \
            --conc_model "$conc_model" \
            --abs_temperature "$abs_temp" \
            --conc_temperature "$conc_temp" \
            --embedding_threshold -1.0 \
            --study "$benchmark" \
            --extra_context \
            --output_dir "benchmarks/injec/results/${abs_model}-${conc_model}"
    done
}

# GPT-4o
run_benchmark "gpt-4o-2024-11-20" "gpt-4o-2024-11-20" 0.0 0.0

# GPT-4.1
run_benchmark "gpt-4.1-2025-04-14" "gpt-4.1-2025-04-14" 0.0 0.0

# Claude 3.7 Sonnet
run_benchmark "claude-3-7-sonnet-20250219" "claude-3-7-sonnet-20250219" 0.0 0.0

# Default config
# note 1.0 temperature needed for o3-mini
run_benchmark "gpt-4o-2024-11-20" "o3-mini-2025-01-31" 0.0 1.0