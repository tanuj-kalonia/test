Scoring Model
=============

This document outlines the steps to score a machine learning model using the `score.py` script.

**Purpose**: The `score.py` script is designed to evaluate the performance of a trained machine learning model.

**Usage**: Execute the script with the trained model file and the input dataset folder path.

Functionality
-------------

The `score_model` function in the script loads the trained model, makes predictions on the input data, and evaluates its performance using appropriate metrics.

- **score_model**:
  - Purpose: Loads the trained model, makes predictions, and evaluates performance.
  - Arguments:
    - `model_path` (str): Path to the trained model file.
    - `data_path` (str): Path to the input dataset folder.
  - Returns: Evaluation metrics.

Logging
-------

The script logs information about the scoring process to a log file located in the "logs" directory.

