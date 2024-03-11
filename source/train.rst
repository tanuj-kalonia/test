Training Model
==============

This document outlines the steps to train a machine learning model using the `train.py` script.

**Purpose**: The `train.py` script is designed to train machine learning models using input data.

**Usage**: Execute the script with the input data folder path and optionally specify the output model folder path.

Functionality
-------------

The `train_model` function in the script loads data, preprocesses it, trains machine learning models, evaluates their performance, and saves the best model.

- **train_model**:
  - Purpose: Loads data, preprocesses, trains models, evaluates performance, and saves the best model.
  - Arguments:
    - `data_path` (str): Path to the input data folder.
    - `model_path` (str, optional): Path to the output model folder.
  - Returns: None

Logging
-------

The script logs information about the training process to a log file located in the "logs" directory.

