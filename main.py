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
plt.figure(figsize=(10, 7))
plt.scatter(data['longitude'], data['latitude'], alpha=0.4,
            s=data['population']/100, c=data['median_house_value'], cmap='jet', label='Population')
plt.colorbar(label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('California Housing Prices')
plt.legend()
plt.show()

## Histogram of all numerical attributes
plt.figure(figsize=(10, 7))
data.hist(bins=50, figsize=(12, 8))
plt.show()

data["income_cat"]= pd.cut(data["median_income"],
                              bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                              labels=[1, 2, 3, 4, 5])

data['income_cat'].value_counts().plot(kind='bar', figsize=(10,6))
plt.xlabel('Income Category')
plt.ylabel('Number of Households')
plt.title('Distribution of Income Categories')
plt.show()

# Split the data into training and testing sets based on income category

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_split=[]
for train_index, test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
    strat_split.append((strat_train_set, strat_test_set))

strat_train_set, strat_test_set = strat_split[0]

strat_train_set.value_counts('income_cat').sort_index().plot(kind='bar', figsize=(10,6))
plt.xlabel('Income Category')
plt.ylabel('Number of Households')
plt.title('Stratified Training Set Income Categories')
plt.show()

print("values count of train set", strat_train_set['income_cat'].value_counts())
print("values count of test set", strat_test_set['income_cat'].value_counts())

strat_train_set.info()
strat_test_set.info()

strat_train_set['income_cat'].head()

# Remove income_cat attribute so the data is back to its original state
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

#explore and visualize data to get better understanding of data relationships

data = strat_train_set.copy()
data.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
          s=data['population']/100, label='Population', figsize=(10,7),
          c=data['median_house_value'], cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()

corr_data = data.drop("ocean_proximity", axis=1)
# Calculate the correlation matrix
corr_matrix = corr_data.corr()
print("\n----------- CORRELATION MATRIX -----------")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Scatter plot to visualize correlation between median_income and median_house_value
data.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Median Income vs. Median House Value')
plt.show()

data.info()

# Prepare the data for machine learning algorithms
data = strat_train_set.drop("median_house_value", axis=1)
data_labels = strat_train_set["median_house_value"].copy()
print("\n----------- DATA PREPARATION -----------")
print("Data prepared for machine learning algorithms.")
data.info()
data_labels.info()

#in this module book gave me inportant advise. I should create a function to do all the preprocessing steps. 
#This is important because when I will be using the model to make predictions on new data, I will need to apply the same preprocessing steps to the new data. 
#By creating a function, I can ensure that the same steps are applied consistently.

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(data):
    # preprocessing the data usibg imputation and scaling
    imputer = SimpleImputer(strategy="median")
    data_num = data.drop("ocean_proximity", axis=1)
    imputer.fit(data_num)
    X = imputer.transform(data_num)
    data_tr = pd.DataFrame(X, columns=data_num.columns, index=data_num.index)

    # #using onehotencoder to build a pipeline for preprocessing the data
    cat_encoder = OneHotEncoder()
    data_cat_1hot = cat_encoder.fit_transform(data[["ocean_proximity"]])
    
    # Feature scaling
    scaler = StandardScaler()
    data_tr_scaled = scaler.fit_transform(data_tr)
    data_tr_scaled = pd.DataFrame(data_tr_scaled, columns=data_tr.columns, index=data_tr.index)

    return data_tr_scaled, data_cat_1hot

print("\n----------- PREPROCESSING DATA -----------", "\n")
data_tr_scaled, data_cat_1hot = preprocess_data(data)
print("Preprocessed numerical data shape:", data_tr_scaled.shape)
print("Preprocessed categorical data shape:", data_cat_1hot.shape)
print("info of original data:", data.info())
print("info of labels:", data_labels.info())

#lets use columntransformer to combine the numerical and categorical data into one dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define the column transformer to apply the appropriate transformations to numerical and categorical features
num_attribs = data.drop("ocean_proximity", axis=1).columns
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])  

# Apply the full pipeline to the training data
data_prepared = full_pipeline.fit_transform(data)
print("\n----------- FULL PIPELINE PREPARATION -----------", "\n")
print("Preprocessed data shape after full pipeline:", data_prepared.shape)

# Now we can train a machine learning model using the preprocessed data. For example, we can use a Random Forest Regressor to predict the median house value.
from sklearn.ensemble import RandomForestRegressor
# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(data_prepared, data_labels)
print("\n----------- MODEL TRAINING -----------", "\n")
print("Model trained successfully.")

predictions = model.predict(data_prepared)
print("\n----------- MODEL PREDICTIONS -----------", "\n")
print("Predictions on training data:", predictions[:5])

# Evaluate the model using Mean Absolute Error
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(data_labels, predictions)
print("\n----------- MODEL EVALUATION -----------", "\n")
print("Mean Absolute Error on training data:", mae)

#use rmse to evaluate the model
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(data_labels, predictions))
print("Root Mean Squared Error on training data:", rmse)

#cross validate the model using cross_val_score
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data_prepared, data_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print("\n----------- CROSS-VALIDATION SCORES -----------", "\n")
print("Cross-validation RMSE scores:", rmse_scores)
print("Mean RMSE:", rmse_scores.mean())
print("Standard Deviation of RMSE:", rmse_scores.std()) 
