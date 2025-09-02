# ACE

## Abstract

LLM-integrated app systems extend the utility of
Large Language Models (LLMs) with third-party apps that
are invoked by a system LLM using interleaved planning and
execution phases to answer user queries. These systems introduce
new attack vectors where malicious apps can cause integrity
violation of planning or execution, availability breakdown, or
privacy compromise during execution.

In this work, we identify new attacks impacting the integrity of
planning, as well as the integrity and availability of execution in
LLM-integrated apps, and demonstrate them against IsolateGPT,
a recent solution designed to mitigate attacks from malicious
apps. We propose Abstract-Concrete-Execute (ACE), a
new secure architecture for LLM-integrated app systems that
provides security guarantees for system planning and execution.
Specifically, ACE decouples planning into two phases by first
creating an abstract execution plan using only trusted informa-
tion, and then mapping the abstract plan to a concrete plan
using installed system apps. We verify that the plans generated
by our system satisfy user-specified secure information flow
constraints via static analysis on the structured plan output.
During execution, ACE enforces data and capability barriers
between apps, and ensures that the execution is conducted
according to the trusted abstract plan. We show experimentally
that ACE is secure against attacks from the INJECAGENT and
Agent Security Bench benchmarks for indirect prompt injection,
and our newly introduced attacks. We also evaluate the utility
of ACE in realistic environments, using the Tool Usage suite
from the LangChain benchmark. Our architecture represents a
significant advancement towards hardening LLM-based systems
using system security principles.

## Setup

### 1. Install uv
This project uses [uv](https://docs.astral.sh/uv/) to manage dependencies and run the `ace` module with an OpenAI model.
Follow the instructions here: [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Verify installation:
```bash
uv --version
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure your API key
Create a .env file in the project root and add your OpenAI key

```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

## Interactive Usage

To run in **interactive mode**, use:

```bash
uv run python -m ace
```

## Experiments

### Case-Studies
To interactively pick which case study (1–3) to run.

``` bash
./benchmarks/case_studies/run.sh 
```

### Langchain

Install benchmark dependencies: 

```bash
uv sync --group bench-langchain
```

To reproduce all configurations

```bash
./benchmarks/langchain/run_all.sh
```

To reproduce a specific configuration

```bash
uv run python -m benchmarks.langchain.run \
    --abs_model "gpt-4.1-2025-04-14" \
    --conc_model "gpt-4.1-2025-04-14" \
    --abs_temperature 0.0 \
    --conc_temperature 0.0 \
    --embedding_threshold -1.0 \
    --benchmark_name "typewriter1" \
    --extra_context
```


### ASB

Install benchmark dependencies: 

```bash
uv sync --group bench-asb
```

To reproduce all configs

```bash
./benchmarks/asb/run_all.sh
```

To reproduce a specific configuration

```bash
uv run python -m benchmarks.asb.run \
    --abs_model "gpt-4.1-2025-04-14" \
    --conc_model "gpt-4.1-mini-2025-04-14" \
    --embedding_threshold 0.0 \
    --extra_context \
    --short

```

To reproduce results

```bash
uv run python benchmarks.asb.analyze --results_json <path_to_results_file>
```

### InjecAgent

Install benchmark dependencies: 

```bash
uv sync --group bench-injec
```

To reproduce all configs

```bash
./benchmarks/injec/run_all.sh
```

To reproduce a specific configuration

```bash

```

To reproduce results

```bash

```

## Citation

Please cite our work as follows for any purpose of usage.

```tex
@misc{li2025acesecurityarchitecturellmintegrated,
      title={ACE: A Security Architecture for LLM-Integrated App Systems}, 
      author={Evan Li and Tushin Mallick and Evan Rose and William Robertson and Alina Oprea and Cristina Nita-Rotaru},
      year={2025},
      eprint={2504.20984},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2504.20984}, 
}
```