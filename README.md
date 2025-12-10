# Transformers

This repository contains the resources to follow the Transformers talk at PyDay. The commits are organized in the same order as the talk. If you wish to observe the architecture construction as performed during the talk, you simply need to traverse the commit history chronologically, from oldest to newest, to see the implementation from scratch.

## Dependencies

You must have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

## Installation

1. Download and install the dependencies using conda:

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the dependencies by executing:

    ```bash
    conda activate transformers-pyday
    ```

> If you use VSCode, you likely want the editor to recognize the dependencies you just installed. To do this, run the command `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) and choose the "Select Interpreter" option; then select `transformers-pyday`. This way, VSCode's IntelliSense can recognize dependencies like PyTorch in the code.

## Running the Code

To execute the code, use:

```bash
python main.py
