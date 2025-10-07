# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load your data
issues_df = pd.read_csv('preprocessed_tensorflow_issues.csv', parse_dates=['created_at', 'closed_at'])

# Prepare your data based on EDA insights
# Aggregate issue data monthly
monthly_data = issues_df.groupby([issues_df['created_at'].dt.to_period('M')]).agg(
    monthly_created=('id', 'size'),
    monthly_closed=('state', lambda x: (x == 'closed').sum())
).reset_index()

# Create lagged feature for previous month's closed issues
monthly_data['prev_monthly_closed'] = monthly_data['monthly_closed'].shift(1)

# Drop the first row to remove NaN values for the lagged feature
monthly_data = monthly_data.dropna()

# Define features and target variable
X = monthly_data[['monthly_created', 'prev_monthly_closed']][:-1]  # Exclude last month for which we don't have a target
y = monthly_data['monthly_closed'][1:].values  # Target starts from the second month

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'R-squared: {r2_score(y_test, y_pred)}')

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
print(f'Average CV MSE: {-cv_scores.mean()}')
