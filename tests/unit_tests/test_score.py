import unittest

from house_price_prediction import score


class TestPackagesInstallation(unittest.TestCase):
    def test_model_score(self):
        model_path = "models/final_model.pkl"
        housing_path = "datasets/housing"

        rmse = score.find_model_score(model_path, housing_path)

        self.assertIsInstance(rmse, float)
        self.assertGreater(rmse, 0)


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
