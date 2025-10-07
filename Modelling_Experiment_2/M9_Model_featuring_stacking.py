import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load your data
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

# Drop the first row to avoid NaN values in lagged features
monthly_data.dropna(inplace=True)

# Define features and target variable
X = monthly_data[['monthly_created_issues', 'prev_monthly_closed', 'prev_contributors', 'prev_avg_labels']]
y = monthly_data['monthly_closed_issues']

# Ensure we're predicting for the next month by shifting the target variable
X = X[:-1]
y = y.shift(-1).dropna()

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define base learners
estimators = [
    ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(random_state=42)),
    ('lasso', LassoCV(cv=5, random_state=42, max_iter=10000))
]

# Initialize Stacking Regressor with LinearRegression as meta-learner
stacking_regressor = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression()
)

# Fit the stacking model
stacking_regressor.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = stacking_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')

# Optionally, perform cross-validation with the optimal alpha
cv_scores_optimal = cross_val_score(stacking_regressor, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
print(f'Average CV MSE: {-cv_scores_optimal.mean()}')
