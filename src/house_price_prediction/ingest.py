import argparse
# import logging
import os
import tarfile

import pandas as pd
from six.moves import urllib
from house_price_prediction.setup_logs import setup_logger


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(logging ,housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    logging.info("Data Fetched")
    return


def load_housing_data(logging,housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    if not csv_path:
        logging.error("Not path found")

    else: logging.info("Data Loaded")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    print("start")
    parser = argparse.ArgumentParser("Enter path for storing housing csv")

    parser.add_argument("dataset_path", help="Enter path for string csv file")
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        help='Specify the log level (e.g., DEBUG, INFO)'
    )

    parser.add_argument(
        '--log-path',
        type=str,
        default="logs/program_logs.txt",
        help='Specify the log file path'
    )

    parser.add_argument(
        '--no-console-log',
        action='store_true',
        help='Disable logging to console'
    )

    args = parser.parse_args()
    path = args.dataset_path.split("/")

    housing_path = os.path.join(path[0], path[1])

    # setup_logging('DEBUG', TRUE: log to a file, 'logs/program_log.txt',False: no to console)

    logging = setup_logger("ingest_loger",args.log_path, args.log_level,args.no_console_log)

    logging.info("Ingestion start")
    fetch_housing_data(logging, HOUSING_URL, housing_path)
    logging.info("Ingestion end")
    print("end")
