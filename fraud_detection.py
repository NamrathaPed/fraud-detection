
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Load the dataset
customer_df = pd.read_csv(".\ecommerce fraud data\Customer_DF (1).csv")
print(customer_df)
transaction_df = pd.read_csv(".\ecommerce fraud data\cust_transaction_details (1).csv")
print(transaction_df)

# Merge datasets
merged_df = pd.merge(customer_df, transaction_df, on='customerEmail', how='inner')  # Adjust column name as needed

# Display dataset overview
print("Merged Dataset Shape:", merged_df.shape)
print(merged_df.head())

print("Columns in merged_df:", merged_df.columns)


print("Missing Values:\n", merged_df.isnull().sum())
merged_df.fillna(0, inplace=True)  # Replace NaN values with 0 (or use other strategies)
#visualize fraud ditribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Fraud', data=merged_df)
plt.title("Fraud vs Non-Fraud Cases")
plt.show()

# List of columns to drop
columns_to_drop = [
    'Unnamed: 0_x', 'customerEmail', 'customerPhone', 'customerDevice',
    'customerIPAddress', 'customerBillingAddress', 'Unnamed: 0_y'
]

# Dropping the unnecessary columns
merged_df_cleaned = merged_df.drop(columns=columns_to_drop, axis=1)
print("Columns after dropping unnecessary ones:")
print(merged_df_cleaned.columns)

#correlation heatmap
numeric_df = merged_df_cleaned.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Define features and target
X = merged_df_cleaned.drop('Fraud', axis=1)  # Features
y = merged_df_cleaned['Fraud']              # Target (Fraud or Not Fraud)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from pandas import get_dummies

# One-hot encode categorical features
X_train = get_dummies(X_train)
X_test = get_dummies(X_test)

# Align columns for consistency (fill missing columns with zeros)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print(X_train.columns)
print(X_test.columns)

# Initialize the model
model = RandomForestClassifier(random_state=42)
# Train the model
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Get feature importances
feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)

# Visualize top 10 features
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances")
plt.show()


# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],               # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],             # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],             # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],               # Minimum samples required to form a leaf
    'bootstrap': [True, False]                   # Whether bootstrap sampling is used
}

# Initialize the RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, 
                                   n_iter=50, cv=3, n_jobs=-1, 
                                   scoring='accuracy', random_state=42, verbose=2)

# Fit the randomized search to the data
random_search.fit(X_train, y_train)

# Best hyperparameters
print("Best Hyperparameters from Randomized Search:", random_search.best_params_)

# Best score
print("Best Accuracy Score from Randomized Search:", random_search.best_score_)

# Use the best model from the search
best_model = random_search.best_estimator_  # Or random_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate performance
print("Final Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




