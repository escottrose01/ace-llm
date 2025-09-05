import argparse
import json
from pathlib import Path

import pandas as pd


def main(args: argparse.Namespace):
    dh_path = args.result_path / "directharm" / "results.jsonl"
    ds_path = args.result_path / "datastealing" / "results.jsonl"

    # Read results
    with open(dh_path) as f:
        data_dh = [json.loads(line) for line in f if line.strip()]
    with open(ds_path) as f:
        data_ds = [json.loads(line) for line in f if line.strip()]

    # Build DataFrame for scores
    df_dh = pd.DataFrame(
        [
            {
                "Study": "directharm",
                "Security Score": entry.get("Security Score", 0),
                "Utility Score": entry.get("Utility Score", 0),
                "Plan Time": entry.get("Trace", {}).get("Plan Time", 0),
                "Match Time": entry.get("Trace", {}).get("Match Time", 0),
                "Execution Time": entry.get("Trace", {}).get("Execution Time", 0),
            }
            for entry in data_dh
        ]
    )
    df_ds = pd.DataFrame(
        [
            {
                "Study": "datastealing",
                "Security Score": entry.get("Security Score", 0),
                "Utility Score": entry.get("Utility Score", 0),
                "Plan Time": entry.get("Trace", {}).get("Plan Time", 0),
                "Match Time": entry.get("Trace", {}).get("Match Time", 0),
                "Execution Time": entry.get("Trace", {}).get("Execution Time", 0),
            }
            for entry in data_ds
        ]
    )

    df = pd.concat([df_dh, df_ds], ignore_index=True)

    # Mean scores per study
    df_grouped = df.groupby("Study").mean()

    # Compute overall average scores
    overall = df.drop("Study", axis=1).mean().to_frame().T
    overall.index = ["Overall"]  # type: ignore
    df_grouped = pd.concat([df_grouped, overall])

    # Collect timing info from successful runs
    statuses = [d.get("Trace", {}).get("Status", "UNKNOWN") for d in data_dh + data_ds]
    status_counts = pd.Series(statuses).value_counts().to_dict()

    # Build status DataFrame
    df_status = pd.Series(status_counts, name="Count").rename_axis("Status").reset_index()
    total = df_status["Count"].sum()
    df_status["Percent"] = (df_status["Count"] / total * 100).round(2)
    df_status = df_status.sort_values(["Count", "Status"], ascending=[False, True])

    # Compute phase success rates
    def counts(entries):
        traces = [e.get("Trace", {}) for e in entries]
        abstract = sum(1 for t in traces if "Abstract Plan" in t)
        concrete = sum(1 for t in traces if "Tool Mapping" in t)
        ok = sum(1 for t in traces if t.get("Status") == "OK")
        return abstract, concrete, ok

    a_dh, c_dh, ok_dh = counts(data_dh)
    a_ds, c_ds, ok_ds = counts(data_ds)
    a_all, c_all, ok_all = a_dh + a_ds, c_dh + c_ds, ok_dh + ok_ds

    df_grouped.loc["directharm", "Matching Succ"] = c_dh / a_dh
    df_grouped.loc["directharm", "Execution Succ"] = ok_dh / c_dh
    df_grouped.loc["datastealing", "Matching Succ"] = c_ds / a_ds
    df_grouped.loc["datastealing", "Execution Succ"] = ok_ds / c_ds
    df_grouped.loc["Overall", "Matching Succ"] = c_all / a_all
    df_grouped.loc["Overall", "Execution Succ"] = ok_all / c_all

    # Display analysis
    print(df_grouped.to_string())
    print()
    print(df_status.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze InjecAgent benchmark results.")
    parser.add_argument("result_path", type=Path, help="Path to the results directory")
    args = parser.parse_args()

    main(args)
