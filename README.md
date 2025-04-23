# ACE

## Setup Instructions

1. **Create a Python environment** using your preferred environment manager (e.g., conda, pyenv, virtualenv):

```bash
conda create -n your-env-name python=3.10
conda activate your-env-name
```

2. **Install the package in editable mode**:

```bash
pip install -e .
```

3. **Set up your OpenAI API key**:

Create a `.env` file in the root of the repository and add the following line:

```
OPENAI_API_KEY=sk-...
```

4. **Build Docker containers**:

Navigate to the `sentinel/execute` directory and build the necessary containers:

```bash
cd sentinel/execute
docker build -t worker-image -f Dockerfile.worker .
docker build -t tool-runner-image -f Dockerfile.tool .
cd ../../
```

## Usage

- To run in **interactive mode**, use:

```bash
python sentinel/main.py
```

- To **reproduce `injecAgent` results**, use:

```bash
python injec_eval/injec.py
```

## Notes

- The default model used is `gpt-4o-mini-2024-07-18`.  
  To change the model, locate the relevant code section and modify the model identifier.