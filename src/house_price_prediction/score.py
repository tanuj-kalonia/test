import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from house_price_prediction.ingest import load_housing_data
from house_price_prediction.setup_logs import setup_logger

def evaluate_model(logging,model, housing_prepared, price_labels):
    print(model)
    try:
        prediction = model.predict(housing_prepared)
        mse = mean_squared_error(price_labels, prediction)
        rmse = np.sqrt(mse)
        return rmse
    except Exception as e:
        logging.info("Error :", e)


def find_model_score(logging,model_path, housing_path):
    try:
        with open(model_path, "rb") as f:
            final_model = pickle.load(f)

        housing = load_housing_data(logging,housing_path)
        housing_price_labels = housing["median_house_value"].copy()
        housing = housing.drop("median_house_value", axis=1)

        imputer = SimpleImputer(strategy="median")
        housing_num = housing.drop("ocean_proximity", axis=1)
        imputer.fit(housing_num)
        X = imputer.transform(housing_num)

        housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
        housing_cat = housing[["ocean_proximity"]]
        housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
        root_mean_sqr_error = evaluate_model(
            logging,final_model, housing_prepared, housing_price_labels
        )
        print(f"Root Mean Square Error : {root_mean_sqr_error}")
        logging.info("Evaluated model successfully.")

        return root_mean_sqr_error

    except Exception as e:
        logging.error("Error :", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Takes the model and return its score."
    )
    parser.add_argument(
        "model_path", type=str, help="Enter the path to the model.pkl file"
    )
    parser.add_argument(
        "housing_path", type=str, help="Enter the path to the housing.csv file"
    )
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
    model_path = args.model_path
    housing_path = args.housing_path

    logging = setup_logger("Score_loger",args.log_path, args.log_level,args.no_console_log)
    find_model_score(logging,model_path, housing_path)

    logging.info("Initiated execution of score.py file.")

