# California Housing Price Prediction 🏠

An end-to-end Machine Learning project based on Chapter 2 of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* by Aurélien Géron. This project builds a regression model to predict median house values in Californian districts using 1990 census data.

## 📌 Project Overview
The goal is to predict the median house value (`median_house_value`) for a given district (block group). This serves as a proxy for a downstream machine learning system within a real estate investment pipeline.

**Key Components:**
* **Data Cleaning:** Handling missing values and outliers.
* **Feature Engineering:** Creating custom attributes like `rooms_per_household`.
* **Pipeline Construction:** Using `Pipeline` and `ColumnTransformer` to automate preprocessing.
* **Model Selection:** Comparing Linear Regression, Decision Trees, and Random Forests.
* **Evaluation:** Using RMSE (Root Mean Square Error) and Cross-Validation.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:**
    * `pandas` & `numpy` (Data Manipulation)
    * `matplotlib` & `seaborn` (Visualization)
    * `scikit-learn` (Modeling, Preprocessing, Pipelines)

## 📂 Dataset
The dataset is based on 1990 California Census data. It includes metrics such as:
* Longitude/Latitude
* Housing Median Age
* Total Rooms/Bedrooms
* Population
* Households
* Median Income
* Ocean Proximity

*Source: The dataset is fetched dynamically inside the notebook or can be found at the [StatLib repository](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).*

## 🚀 Key Techniques Implemented

### 1. Stratified Sampling
Instead of a random train/test split, I utilized `StratifiedShuffleSplit` on the **Median Income** category. This ensures the test set is representative of the income distribution in the full dataset, preventing sampling bias which is critical for this dataset.

### 2. Custom Transformers
Implemented custom Scikit-Learn transformers to integrate seamlessly into pipelines:
* `CombinedAttributesAdder`: Automatically generates new features:
    * *Rooms per Household*
    * *Population per Household*
    * *Bedrooms per Room*

### 3. Transformation Pipelines
Data preprocessing is handled via a unified `ColumnTransformer`:
* **Numerical Data:** Imputation (Median), Attribute Addition, Standardization (`StandardScaler`).
* **Categorical Data:** One-Hot Encoding (`OneHotEncoder`).

## 📊 Results (Example)
* **Linear Regression RMSE:** ~$68,000 (Underfitting)
* **Decision Tree RMSE:** ~$0.0 (Overfitting)
* **Random Forest RMSE:** ~$50,000 (Best performance)

*Note: Final model fine-tuning was performed using `GridSearchCV`.*

## 💻 How to Run
1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/california-housing-ml.git](https://github.com/YOUR_USERNAME/california-housing-ml.git)
    cd california-housing-ml
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib scikit-learn jupyter
    ```

3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open `California_Housing.ipynb` and run the cells.

## 🤝 Acknowledgments
* Dataset originally from the 1990 California Census.
* Project structure guided by Aurélien Géron's *Hands-On Machine Learning* book.

---
**Author:** [Your Name]
