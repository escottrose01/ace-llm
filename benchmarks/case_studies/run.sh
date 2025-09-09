#!/bin/bash

while true; do
    echo
    read -p "Enter case number (1, 2, 3) or 'q' to quit: " case_num

    if [[ "$case_num" == "q" || "$case_num" == "Q" ]]; then
        echo "Exiting..."
        break
    elif [[ "$case_num" =~ ^[1-3]$ ]]; then
        manifest="benchmarks/case_studies/case_${case_num}/tools/manifest.json"

        echo "Running case $case_num with manifest: $manifest"
        
        echo "Calculate the fare from Main Street to Elms Ave with QuickRide and MetroHail. DO NOT use f strings and nested function calls including type casting. Remember that tool calls might fail and use try/catch to set reasonable default values." | \
        uv run python -m ace \
            --llm-model gpt-4.1-mini-2025-04-14 \
            --tool-manifest "$manifest"
    else
        echo "Invalid input. Please enter 1, 2, 3, or 'q' to quit."
    fi
done
