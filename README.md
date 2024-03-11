# Median housing value prediction

The housing data was downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/.
The script has codes to download the data. We have modelled the median house value on given housing data.

The following techniques have been used:

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - A proper folder structure was created as per the standard coding practices.
 - The code was seggreegated into ingest_data, train and score files.
   - ingest_data.py fetched the housing data from the web and stored it in a CSV file.
   - train.py trained the dataset, applying the aforementioned techniques, and generated a model.pkl file
   - score.py evaluated the model based on the root mean square error value.
 - Testing was done to ensure proper functioning of the project.
   - functional tests - checked for the proper installation of the libraries and packages required for the project.
   - unit tests - checked for the proper execution of the ingest_data, train and score files.
 - The documentation of the whole project was generated using Sphinx.

## To excute the script
python src/HousePricePrediction/main.py (Assuming the script is executing in the home directory.)

## Instructions to run the code

A new conda environment mle-dev was created to run this particular script.
Some minor bugs were fixed from the script.
The code was formatted using black, imports were sorted using isort and the code was linted using flake8 libraries.
After the script ran successfully, we exported the conda environment into an env.yml file.
bash

# To create an environment from the env.yml file
```
conda env create -f env.yml
```

# Black was used to properly format the code
```
black nonstandardcode.pybash
```

# isort was used to sort the imports
```
isort nonstandardcode.py
```

# flake8 was used for code linting (the setup.cfg file contains custom options that would be used by flake8)
```
flake8 nonstandardcode.pybash
```

# To activate the environment
```
conda activate mle-dev
```

# To execute script
```
python3 src/HousePricePrediction/test.py
```