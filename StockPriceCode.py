import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Step 1: Load data from the zipped CSV file
df = pd.read_csv('stock prices.csv.zip', compression='zip')

# Step 2: Display the first few rows to understand its structure
print(df.head())

# Step 3: Check for missing values
print("Missing values before imputation:")
print(df.isnull().sum())

# Step 4: Select numeric columns for imputation and model training
numeric_columns = ['open', 'high', 'low', 'close']
df_numeric = df[numeric_columns]

# Step 5: Impute missing values with the mean for numeric columns
imputer = SimpleImputer(strategy='mean')
df_numeric_filled = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)

# Step 6: Check again for missing values (should be zero now)
print("\nMissing values after imputation:")
print(df_numeric_filled.isnull().sum())

# Step 7: Reconstruct the DataFrame with imputed values
df_filled = pd.concat([df_numeric_filled, df[df.columns.difference(numeric_columns)]], axis=1)

# Step 8: Selecting features and target variable
X = df_filled[['open', 'high', 'low', 'close']]
y = df_filled['close']

# Step 9: Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 10: Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Making predictions
y_pred = model.predict(X_test)

# Step 12: Evaluating the model
print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Step 13: Plotting actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Line')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
