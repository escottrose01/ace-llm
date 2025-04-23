from dotenv import load_dotenv
import argparse
import logging
import json

from langchain_openai import ChatOpenAI

from agent import InjecSentinelAgent
from injec_eval import evaluate, get_all_user_cases, test_tools

# from sentinel.prompts.concrete_templates import new_conc_compat_template

# To re-record all InjecAgent tools in schema_db or injec_manifest.jsonl, use this function
from util.injec_data import fill_manifest, fill_schema

load_dotenv()


def main():
    result = evaluate(start=0, end=30, direct_harm=0)
    print("---------- RESULTS -----------")
    print(json.dumps(result, indent=4))

    counter = {
        "Utility Success": 0,
        "Matching Failure": 0,
        "Execution Failure": 0,
        "Security Success": 0,
        "Security Failure": 0
    }

    for item in result:
        if item["Utility Result"] == "Success":
            counter["Utility Success"] += 1
        else:
            if item["Failure Type"] == "Matching":
                counter["Matching Failure"] += 1
            elif item["Failure Type"] == "Execution":
                counter["Execution Failure"] += 1
            else:
                raise RuntimeError("Failure not found")
        if item["Defense Result"] == "Success":
            counter["Security Success"] += 1
        else:
            counter["Security Failure"] += 1
    print(counter)


def calculate_metrics(ls: list[dict]):
    try:
        sums = {'Utility Success': 0, 'Matching Failure': 0,
                'Execution Failure': 0, 'Security Success': 0, 'Security Failure': 0}
        for item in ls:
            for key in item:
                sums[key] += item[key]
        total = (sums['Security Success'] + sums["Security Failure"])

        usr = sums['Utility Success'] / total
        msr = 1 - (sums['Matching Failure'] / total)
        esr = 1 - (sums['Execution Failure'] / (total * msr))

        err = abs(usr - (1.00 * msr * esr))
        results = {'tests': sums, 'Utility Success Rate': usr, 'Matching Success Rate': msr,
                   "Execution Success Rate": esr, "Calculated Error": err}
        return results

    except Exception as e:
        raise ValueError(
            "caluclate_metrics must be called with a list of dictionaries with correct keys", str(e))


if __name__ == "__main__":
    main()
