import os
import unittest

import pandas as pd

from house_price_prediction import ingest, train


class TestPackageInstallation(unittest.TestCase):
    # Check wheter the model/model_pkl exists or not after training
    def test_model_trainig(self):
        housing_df = ingest.load_housing_data("datasets/housing")
        self.assertIsInstance(housing_df, pd.DataFrame)

        # check the output folder
        model_folder = "models"
        os.makedirs(model_folder, exist_ok=True)
        train.model_train(housing_df)

        model_path = os.path.join(model_folder, "final_model.pkl")

        # check model_path exists or not
        self.assertTrue(os.path.exists(model_path))


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
