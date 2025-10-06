import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import joblib

FIG_DIR = Path("figures")
OUT_DIR = Path("outputs")
FIG_DIR.mkdir(exist_ok=True, parents=True)
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Load data
commits_df = pd.read_csv('preprocessed_tensorflow_commits.csv', parse_dates=['date'])
issues_df = pd.read_csv('preprocessed_tensorflow_issues.csv', parse_dates=['created_at', 'closed_at'])

# Prepare additional features
commits_df['month_year'] = commits_df['date'].dt.to_period('M')
issues_df['month_year'] = issues_df['created_at'].dt.to_period('M')

# Number of unique contributors per month
contributors_per_month = commits_df.groupby('month_year')['contributor'].nunique().reset_index()
contributors_per_month.columns = ['month_year', 'contributors_per_month']

# Average labels count per issue per month
avg_labels_per_issue = issues_df.groupby('month_year')['labels_count'].mean().reset_index()
avg_labels_per_issue.columns = ['month_year', 'avg_labels_count']

# Aggregate monthly data from issues
monthly_issues = issues_df.groupby('month_year').agg(
    monthly_created_issues=('id', 'size'),
    monthly_closed_issues=('state', lambda x: (x == 'closed').sum())
).reset_index()

# Merge additional features
monthly_data = pd.merge(monthly_issues, contributors_per_month, on='month_year', how='left')
monthly_data = pd.merge(monthly_data, avg_labels_per_issue, on='month_year', how='left')

# Fill missing values that might result from the merge (if any)
monthly_data.fillna(0, inplace=True)

# Create lagged features for the previous month's data
monthly_data['prev_monthly_closed'] = monthly_data['monthly_closed_issues'].shift(1)
monthly_data['prev_contributors'] = monthly_data['contributors_per_month'].shift(1)
monthly_data['prev_avg_labels'] = monthly_data['avg_labels_count'].shift(1)

# Advanced Temporal Feature Engineering
monthly_data['date'] = pd.to_datetime(monthly_data['month_year'].astype(str))
monthly_data['quarter'] = monthly_data['date'].dt.quarter
monthly_data['year'] = monthly_data['date'].dt.year
monthly_data['month_sin'] = np.sin(2 * np.pi * monthly_data['date'].dt.month / 12)
monthly_data['month_cos'] = np.cos(2 * np.pi * monthly_data['date'].dt.month / 12)

# Drop the first row to avoid NaN values in lagged features and 'date' column as it's no longer needed
monthly_data.dropna(inplace=True)
monthly_data.drop(columns=['date'], inplace=True)

# Define features including new temporal features and target variable
X = monthly_data[['monthly_created_issues', 'prev_monthly_closed', 'prev_contributors', 'prev_avg_labels', 'quarter', 'year', 'month_sin', 'month_cos']]
y = monthly_data['monthly_closed_issues']

# Ensure we're predicting for the next month by shifting the target variable
X = X[:-1]
y = y.shift(-1).dropna()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Use a wider range of alpha values to find the optimal alpha with more regularization
lasso_cv = LassoCV(alphas=np.logspace(-6, 1, 100), cv=10, random_state=42, max_iter=100000)
lasso_cv.fit(X_train, y_train)

# Train a new Lasso model with the new optimal alpha
lasso_optimal = Lasso(alpha=lasso_cv.alpha_, max_iter=100000)
lasso_optimal.fit(X_train, y_train)

# Make predictions and evaluate the model with the optimal alpha
y_pred_optimal = lasso_optimal.predict(X_test)
mse_optimal = mean_squared_error(y_test, y_pred_optimal)
r2_optimal = r2_score(y_test, y_pred_optimal)

print(f'Optimal alpha: {lasso_cv.alpha_}')
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred_optimal)}')
print(f'R-squared: {r2_score(y_test, y_pred_optimal)}')

# Optionally, perform cross-validation with the optimal alpha
cv_scores_optimal = cross_val_score(lasso_optimal, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
print(f'Average CV MSE: {-cv_scores_optimal.mean()}')

# Plot actual vs predicted values
#plt.scatter(y_test, y_pred_optimal)
#plt.xlabel('Actual Closed Issues Next Month')
#plt.ylabel('Predicted Closed Issues Next Month')
#plt.title('Actual vs Predicted')
#plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#plt.show()

# Calculate residuals
residuals = y_test - y_pred_optimal

# Plot residuals
#plt.scatter(y_test, residuals)
#plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red', linestyles='--')
#plt.xlabel('Actual Closed Issues Next Month')
#plt.ylabel('Residuals')
#plt.title('Residual Plot')
#plt.show()

# Check the distribution of residuals
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=20, rwidth=0.7, color='black')
plt.xlabel('Residual')
plt.ylabel('Frequency')
#plt.title('Residual Distribution')
plt.tight_layout()
plt.grid(axis='y', linestyle='--')
plt.savefig(FIG_DIR / 'Figure_6.svg', format='svg')
plt.close()

# Save the model to disk
joblib.dump(lasso_cv, 'predict_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
