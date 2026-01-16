import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import os

#load dataset and take a quick look at data statistics
# Load dataset
CSV_FILE_PATH = "/Users/pritpatel/Documents/End to End Machine Learning project./CaliforniaHousing/housing.csv"
if os.path.exists(CSV_FILE_PATH):
    data = pd.read_csv(CSV_FILE_PATH)
    print("Dataset loaded successfully.")
    print("First 5 rows of the dataset:", data.head())
else:
    print(f"File not found: {CSV_FILE_PATH}")
    data = pd.DataFrame()  # Empty DataFrame as fallback
pd.set_option('display.max_columns', None) # Adjust the width as needed
    

# get insights of the data 
# 1. info() gives you data types and tells you where the nulls (missing values) are
print("\n----------- DATA INFO -----------")
print(data.info())

# 2. describe() gives you the math (mean, min, max, std) for the numbers
print("\n----------- STATISTICAL SUMMARY -----------")
print(data.describe())

# 3. Only look at value_counts for the CATEGORICAL column (ocean_proximity)
print("\n----------- OCEAN PROXIMITY COUNTS -----------")
print(data["ocean_proximity"].value_counts())

#data visualization

## Scatter plot of latitude vs longitude, colored by median house value
plt.figure(figsize=(10, 6))
plt.scatter(data['longitude'], data['latitude'], alpha=0.4,
            s=data['population']/100, c=data['median_house_value'], cmap='jet', label='Population')
plt.colorbar(label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('California Housing Prices')
plt.legend()
plt.show()

## Histogram of all numerical attributes
plt.figure(figsize=(20, 15))
data.hist(bins=50, figsize=(20,15))
plt.show()

data["income_cat"]= pd.cut(data["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])

data['income_cat'].value_counts().plot(kind='bar', figsize=(10,6))
plt.xlabel('Income Category')
plt.ylabel('Number of Households')
plt.title('Distribution of Income Categories')
plt.show()

#train test split using stratified sampling based on income category
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
print("\n----------- STRATIFIED TRAINING SET INFO -----------")
print(strat_train_set['income_cat'].value_counts() / len(strat_train_set))
print("\n----------- STRATIFIED TEST SET INFO -----------")
print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))

# Remove income_cat attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# Prepare the data for machine learning algorithms
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity", axis=1)
housing_cat = housing[["ocean_proximity"]]
housing_cat_encoded = pd.get_dummies(housing_cat, drop_first=True)
housing_prepared = pd.concat([housing_num, housing_cat_encoded], axis=1)
