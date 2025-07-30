import argparse
import json
from pathlib import Path

import pandas as pd


def format_time(mean, std):
    if pd.isna(std):
        return f"{mean:.2f} ± nan"
    return f"{mean:.2f} ± {std:.2f}"


def main(args: argparse.Namespace):
    def print_table_with_separator(df, title=None, separator_label="Overall"):
        table_str = df.to_string()
        lines = table_str.splitlines()
        if len(lines) <= 2:
            if title:
                print(title)
            print(table_str)
            return
        header = lines[0]
        data_lines = lines[1:]
        main_lines = data_lines[:-1]
        overall_line = data_lines[-1]
        separator = "-" * len(header)
        # Print centered title if provided
        if title:
            print(title.center(len(header)))
        # Print header and data
        print(header)
        for line in main_lines:
            print(line)
        print(separator)
        print(overall_line)
        print()

    # Read results
    with open(args.result_json) as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Build DataFrame for scores
    df_scores = pd.DataFrame(
        [
            {
                "Agent": entry.get("Agent"),
                "Security Score": entry.get("Security Score", 0),
                "Hard Utility Score": entry.get("Hard Utility Score", 0),
                "Soft Utility Score": entry.get("Soft Utility Score", 0),
                "Tool Use Score": entry.get("Tool Use Score", 0),
            }
            for entry in data
            if "Agent" in entry
        ]
    )

    # Mean scores per agent
    grouped_scores = df_scores.groupby("Agent").mean()

    # Compute overall average scores
    overall_scores = grouped_scores.mean().to_frame().T
    overall_scores.index = ["Overall"]  # type: ignore

    # Combine per-agent and overall
    scores_output = pd.concat([grouped_scores, overall_scores])

    # Collect timing info from successful runs
    timing_data = []
    status_counts = {}

    for entry in data:
        trace = entry.get("Trace", {})
        status = trace.get("Status")
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if status == "OK":
            timing_data.append(
                {
                    "Agent": entry.get("Agent"),
                    "Plan Time": trace.get("Plan Time", 0),
                    "Match Time": trace.get("Match Time", 0),
                    "Execution Time": trace.get("Execution Time", 0),
                }
            )

    df_timing = pd.DataFrame(timing_data)

    # Mean and std per agent
    stats = df_timing.groupby("Agent").agg(
        Plan_Time_Mean=("Plan Time", "mean"),
        Plan_Time_Std=("Plan Time", "std"),
        Match_Time_Mean=("Match Time", "mean"),
        Match_Time_Std=("Match Time", "std"),
        Execution_Time_Mean=("Execution Time", "mean"),
        Execution_Time_Std=("Execution Time", "std"),
    )

    # Format timings
    stats["Plan Time"] = stats.apply(lambda row: format_time(row["Plan_Time_Mean"], row["Plan_Time_Std"]), axis=1)
    stats["Match Time"] = stats.apply(lambda row: format_time(row["Match_Time_Mean"], row["Match_Time_Std"]), axis=1)
    stats["Execution Time"] = stats.apply(
        lambda row: format_time(row["Execution_Time_Mean"], row["Execution_Time_Std"]), axis=1
    )

    timing_formatted = stats[["Plan Time", "Match Time", "Execution Time"]]

    # Compute overall timing averages
    overall_timing = df_timing.agg(
        {"Plan Time": ["mean", "std"], "Match Time": ["mean", "std"], "Execution Time": ["mean", "std"]}
    )
    overall_row = pd.DataFrame(
        {
            "Plan Time": [format_time(overall_timing.loc["mean", "Plan Time"], overall_timing.loc["std", "Plan Time"])],
            "Match Time": [
                format_time(overall_timing.loc["mean", "Match Time"], overall_timing.loc["std", "Match Time"])
            ],
            "Execution Time": [
                format_time(overall_timing.loc["mean", "Execution Time"], overall_timing.loc["std", "Execution Time"])
            ],
        },
        index=["Overall"],
    )

    # Combine per-agent and overall
    timing_output = pd.concat([timing_formatted, overall_row])

    # Status breakdown
    status_df = pd.DataFrame.from_dict(status_counts, orient="index", columns=["Count"]).rename_axis("Status")

    # Print per-agent breakdown with separator
    print_table_with_separator(scores_output, title="Average Security and Utility Scores by Agent")
    print_table_with_separator(timing_output, title="Average Runtime for Successful Runs")
    print("Status Breakdown:")
    print(status_df.to_string())

    # Display security failures
    failures = [e for e in data if e.get("Security Score", 1) == 0]

    if failures:
        records = []
        for entry in failures:
            agent = entry.get("Agent")
            attacker = entry.get("Attacker Tool")
            trace = entry.get("Trace", {})
            mapping = trace.get("Tool Mapping", {})

            # Find abstract tools mapped to this attacker
            abstracts = [abs_name for abs_name, info in mapping.items() if info.get("name") == attacker]

            for abstract in abstracts:
                records.append({"Agent": agent, "Abstract Tool": abstract, "Attacker Tool": attacker})

        df_fail = pd.DataFrame(records)
        print("\nSecurity Failures")
        print(df_fail.to_string(index=False))
    else:
        print("\nNo security failures (Security Score == 0) found.")


def _main(args: argparse.Namespace):
    # Read results
    with open(args.result_json) as f:
        data = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(
        [
            {
                "Agent": entry.get("Agent"),
                "Security Score": entry.get("Security Score", 0),
                "Utility Score": entry.get("Utility Score", 0),
                "Tool Use Score": entry.get("Tool Use Score", 0),
            }
            for entry in data
            if "Agent" in entry
        ]
    )

    # Group by agent and compute mean scores
    grouped = df.groupby("Agent").mean().reset_index()

    # Collect timing info from successful runs
    timing_data = []
    status_counts = {}

    for entry in data:
        trace = entry.get("Trace", {})
        status = trace.get("Status")
        if status:
            status_counts[status] = status_counts.get(status, 0) + 1
        if status == "OK":
            timing_data.append(
                {
                    "Agent": entry.get("Agent"),
                    "Plan Time": trace.get("Plan Time", 0),
                    "Match Time": trace.get("Match Time", 0),
                    "Execution Time": trace.get("Execution Time", 0),
                }
            )

    timing_df = pd.DataFrame(timing_data)

    # Compute mean and std
    timing_stats = (
        timing_df.groupby("Agent")
        .agg(
            Plan_Time_Mean=("Plan Time", "mean"),
            Plan_Time_Std=("Plan Time", "std"),
            Match_Time_Mean=("Match Time", "mean"),
            Match_Time_Std=("Match Time", "std"),
            Execution_Time_Mean=("Execution Time", "mean"),
            Execution_Time_Std=("Execution Time", "std"),
        )
        .reset_index()
    )

    timing_stats["Plan Time"] = timing_stats.apply(
        lambda row: format_time(row["Plan_Time_Mean"], row["Plan_Time_Std"]), axis=1
    )
    timing_stats["Match Time"] = timing_stats.apply(
        lambda row: format_time(row["Match_Time_Mean"], row["Match_Time_Std"]), axis=1
    )
    timing_stats["Execution Time"] = timing_stats.apply(
        lambda row: format_time(row["Execution_Time_Mean"], row["Execution_Time_Std"]), axis=1
    )

    timing_formatted = timing_stats[["Agent", "Plan Time", "Match Time", "Execution Time"]]

    # Convert status count dict to DataFrame
    status_df = pd.DataFrame(list(status_counts.items()), columns=["Status", "Count"])

    # Print everything
    print("\nAverage Security and Utility Scores by Agent:")
    print(grouped.to_string(index=False))

    print("\nAverage Runtime (mean ± std) for Successful Runs (Status == 'OK'):")
    print(timing_formatted.to_string(index=False))

    print("\nStatus Breakdown:")
    print(status_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ASB benchmark results.")
    parser.add_argument("result_json", type=Path, help="Path to the results JSON file.")
    args = parser.parse_args()

    main(args)
