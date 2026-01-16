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