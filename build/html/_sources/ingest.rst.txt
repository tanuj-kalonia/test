Ingest Data
===========

This document outlines the steps to ingest data for machine learning using the `ingest_data.py` script.

**Purpose**: The `ingest_data.py` script is designed to preprocess and prepare data for training machine learning models.

**Usage**: Execute the script with the input data folder path and optionally specify the output folder path.

Functionality
-------------

The `ingest_data` function in the script reads raw data, preprocesses it, and saves the processed data to the specified output folder.

- **ingest_data**:
  - Purpose: Reads and preprocesses raw data.
  - Arguments:
    - `input_path` (str): Path to the input data folder.
    - `output_path` (str, optional): Path to the output folder to save processed data.
  - Returns: None

Logging
-------

The script logs information about the data ingestion process to a log file located in the "logs" directory.
