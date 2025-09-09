import argparse
import functools
import json
import operator
from pathlib import Path

import numpy as np
import pandas as pd


def main(args: argparse.Namespace):
    studies = ["directharm", "datastealing"]
    data = dict()
    dfs = []

    for study in studies:
        study_path = args.result_path / study / "results.jsonl"
        if not study_path.exists():
            print(f"Warning: {study_path} does not exist. Skipping.")
            continue
        with open(study_path) as f:
            data[study] = [json.loads(line) for line in f if line.strip()]
    data["Overall"] = functools.reduce(operator.iadd, data.values(), [])

    # Build DataFrame for scores
    for study in data:
        dfs.append(
            pd.DataFrame(
                [
                    {
                        "Study": study,
                        "Security Score": entry.get("Security Score", 0),
                        "Utility Score": entry.get("Utility Score", 0),
                        "Plan Time": entry.get("Trace", {}).get("Plan Time", 0),
                        "Match Time": entry.get("Trace", {}).get("Match Time", 0),
                        "Execution Time": entry.get("Trace", {}).get("Execution Time", 0),
                    }
                    for entry in data[study]
                ]
            )
        )

    df = pd.concat(dfs, ignore_index=True)

    # Mean scores per study
    df_grouped = df.groupby("Study").mean()

    # Collect timing info from successful runs
    statuses = [d.get("Trace", {}).get("Status", "UNKNOWN") for d in data["Overall"]]
    status_counts = pd.Series(statuses).value_counts().to_dict()

    # Build status DataFrame
    df_status = pd.Series(status_counts, name="Count").rename_axis("Status").reset_index()
    total = df_status["Count"].sum()
    df_status["Percent"] = (df_status["Count"] / total * 100).round(2)
    df_status = df_status.sort_values(["Count", "Status"], ascending=[False, True])

    # Compute phase success rates
    for study in data:
        entries = data[study]

        for e in entries:
            u = e.get("User Tool", "")
            t = e.get("Trace", {})
            if e.get("Utility Score", 0) == 0 and u in t.get("Invoked Tools", []):
                print(f"Warning: Utility Score is 0 but User Tool '{u}' was invoked in study '{study}'.")

            at = t.get("Abstract Tools", None)
            if at is not None and len(at) == 0:
                print(f"Warning: Abstract Tools is empty in study '{study}'.")

            if e.get("Utility Score", 0) == 0 and t.get("Status", "") == "OK":
                print("Warning: Utility Score is 0 but overall Status is OK")
                print(t.get("Abstract Plan"))
                print(e.get("Invoked Tools"))

        traces = [e.get("Trace", {}) for e in entries]
        user_tools = [e.get("User Tool", "") for e in entries]

        matched = np.array(
            [
                1 if u in map(lambda x: x.get("name", ""), t.get("Tool Mapping", {}).values()) else 0
                for t, u in zip(traces, user_tools)
            ]
        )
        executed = matched * np.array([1 if t.get("Status") == "OK" else 0 for t in traces])

        # Apparently there was a bug where status would report "OK" but script failed to run (?)
        # So we also check that the list of invoked tools is nonempty
        executed = executed * np.array([1 if len(t.get("Invoked Tools", [])) > 0 else 0 for t in traces])

        e = executed.sum()
        m = matched.sum()

        df_grouped.loc[study, "Matching Succ"] = m / len(entries) if len(entries) > 0 else 0
        df_grouped.loc[study, "Execution Succ"] = e / m if m > 0 else 0

    # Display analysis
    print(df_grouped.to_string())
    print()
    print(df_status.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze InjecAgent benchmark results.")
    parser.add_argument("result_path", type=Path, help="Path to the results directory")
    args = parser.parse_args()

    main(args)
