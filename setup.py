from setuptools import setup, find_packages

# Package meta-data.
NAME = "sentinel"
DESCRIPTION = "ACE: the Abstract-Concrete-Execute security architecture for LLM-integrated app systems. "
URL = ""
AUTHOR = ""
PYTHON_REQUIRES = ">=3.10.16"
VERSION = "0.1.0"

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=PYTHON_REQUIRES,
    URL=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "pygments",
        "langchain",
        "langchain_community",
        "langchain_openai",
        "langchain_anthropic",
        "faiss-cpu",
        "transformers",
        "tensorflow",
        "docker",
        "astor",
        "pydantic",
    ],
)
