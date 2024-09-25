# mlflow-tutorial
Demo repo to show how to use MLflow with Stable-Baselines 3

## Installation
Install the package with poetry.

## Setup MLflow Tracking Environment
Refer to `docs/setup.md`.

## Examples
`scripts` directory contains the following examples:
- `sklean_example.py`: a skelton of MLflow workflow with a simple sklearn model.
- `sb3_example.py`: an example to integrate MLflow to a SB3 training workflow.
- `sb3_pred_example.py`: prediction example by retrieving a registered model.