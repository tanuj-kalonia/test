from setuptools import setup

setup(
    # Project Metadata
    name="house_price_prediction",
    description="A sample Python package to predict the housing price",
    author="Tanuj Kalonia",
    author_email="tanuj.m@tigeranalytics.com",
    # packages=['house_price_prediction'],
    # Dependencies
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "six",
    ],
)
