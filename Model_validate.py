import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
FIG_DIR = Path("figures_test_2")
OUT_DIR = Path("outputs_test_2")
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

# Group by month and count the number of closed issues
issues_df['month'] = issues_df['closed_at'].dt.to_period('M')
closed_issues_per_month = issues_df[issues_df['state'] == 'closed'].groupby('month').size()

# Aggregate monthly data from issues
issues_df['month_year'] = issues_df['closed_at'].dt.to_period('M')
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

# Feature Scaling and Model
scaler = joblib.load('scaler.pkl')
X_scaled = scaler.transform(X)  
model_path = 'predict_model.pkl'
loaded_model = joblib.load(model_path)

# Predict the closed issues using the trained model
# Note: Don't fit the scaler again; use transform only
predicted_closed_issues = loaded_model.predict(X_scaled)

# Ensure the predicted data aligns with the actual data
predicted_closed_issues_series = pd.Series(predicted_closed_issues, index=closed_issues_per_month.index[1:])

# Plot the actual closed issues and predicted closed issues
plt.figure(figsize=(14, 7))  # Adjust the size as needed
plt.plot(closed_issues_per_month.index.astype(str), closed_issues_per_month.values, label='Actual Closed Issues')
plt.plot(predicted_closed_issues_series.index.astype(str), predicted_closed_issues_series.values, label='Predicted Closed Issues', linestyle='--')
plt.xlabel('Month')
plt.ylabel('Number of Issues')
#plt.title('Actual vs Predicted Closed Issues of TensorFlow Over Time')
plt.legend()
plt.xticks(rotation=90)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust the layout to fit the labels
plt.savefig(FIG_DIR / 'Figure_7.svg', format='svg')
plt.show()
