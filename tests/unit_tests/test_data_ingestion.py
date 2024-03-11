import os
import unittest

import pandas as pd

from house_price_prediction.ingest import fetch_housing_data


class TestPackageInstalltion(unittest.TestCase):
    def test_ingest_data(self):
        output_folder = "datasets/housing"
        os.makedirs(output_folder, exist_ok=True)

        fetch_housing_data(housing_path=output_folder)
        dataset_path = os.path.join(output_folder, "housing.csv")

        # check whether the dataset path exists or not
        self.assertTrue(os.path.exists(dataset_path))

        data = pd.read_csv(dataset_path)
        self.assertIsInstance(data, pd.DataFrame)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
