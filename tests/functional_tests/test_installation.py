import importlib.util as imports
import unittest


class Test_Packages_Installation(unittest.TestCase):
    def test_import_numpy(self):
        self.assertTrue(
            imports.find_spec("numpy") is not None, "Numpy is not installed"
        )

    def test_import_pandas(self):
        self.assertTrue(
            imports.find_spec("pandas") is not None, "pandas is not installed"
        )

    def test_import_sklearn(self):
        self.assertTrue(
            imports.find_spec("sklearn") is not None, "sklearn is not installed"
        )

    def test_import_hpp(self):
        self.assertTrue(
            imports.find_spec("house_price_prediction") is not None,
            "hpp is not installed",
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)
