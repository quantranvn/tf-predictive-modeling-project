# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import Ridge
from scipy.stats import uniform
import matplotlib.pyplot as plt

# Load data
commits_df = pd.read_csv('preprocessed_tensorflow_commits.csv', parse_dates=['date'])
issues_df = pd.read_csv('preprocessed_tensorflow_issues.csv', parse_dates=['created_at', 'closed_at'])

# Prepare your time-based features as before
commits_df['Month'] = commits_df['date'].dt.to_period('M')
issues_df['Created_Month'] = issues_df['created_at'].dt.to_period('M')
issues_df['Closed_Month'] = issues_df['closed_at'].dt.to_period('M')
monthly_commits = commits_df.groupby('Month').size()
monthly_created_issues = issues_df.groupby('Created_Month').size()
monthly_closed_issues = issues_df[issues_df['state'] == 'closed'].groupby('Closed_Month').size()

# Merge the datasets and create lagged columns for issues
combined_data = pd.DataFrame({
    'Commits': monthly_commits,
    'CreatedIssues': monthly_created_issues.reindex(monthly_commits.index, fill_value=0),
    'ClosedIssues': monthly_closed_issues.reindex(monthly_commits.index, fill_value=0)
}).fillna(0)

combined_data['ClosedIssuesNextMonth'] = combined_data['ClosedIssues'].shift(-1)

# Drop the last row which will have NaN because of the shift
combined_data.dropna(inplace=True)

# Ensure the DataFrame is not empty before proceeding
if combined_data.empty:
    raise ValueError("The combined_data DataFrame is empty. Check the data processing steps.")

# Define features and target variable from your preprocessed data
X = combined_data[['Commits', 'CreatedIssues', 'ClosedIssues']]
y = combined_data['ClosedIssuesNextMonth']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
ridge_model = Ridge()

# Set up Randomized Search with a wide range
param_distributions = {'alpha': uniform(0.1, 10000)}
random_search = RandomizedSearchCV(ridge_model, param_distributions, random_state=42, n_iter=100, cv=5, scoring='neg_mean_squared_error')

# Perform Random Search
random_search.fit(X_train, y_train)

# Get the best parameters from the Random Search to use in Grid Search
best_params = random_search.best_params_
print(f"Best parameters from Random Search: {best_params}")

# Define a narrower range around the best alpha found
alpha_range = [best_params['alpha'] * 0.5, best_params['alpha'], best_params['alpha'] * 1.5]

# Set up Grid Search within the narrow range
grid_search = GridSearchCV(ridge_model, {'alpha': alpha_range}, cv=5, scoring='neg_mean_squared_error')

# Perform Grid Search
grid_search.fit(X_train, y_train)

# Get the best alpha value from Grid Search
best_alpha = grid_search.best_params_['alpha']
print(f"Optimal alpha value after Grid Search: {best_alpha}")

# Use the best alpha to train the final model
final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train, y_train)

# Make predictions on the test set using the best alpha value
y_pred = final_model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print out the performance metrics
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Plot actual vs predicted values for visual inspection
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Closed Issues Next Month')
plt.ylabel('Predicted Closed Issues Next Month')
plt.title('Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()

# Calculate and plot residuals
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.hlines(y=0, xmin=y_test.min(), xmax=y_test.max(), colors='red', linestyles='--')
plt.xlabel('Actual Closed Issues Next Month')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Check the distribution of residuals
plt.hist(residuals, bins=20)
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.title('Residual Distribution')
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(final_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Print out cross-validation scores
print(f'CV MSE scores: {-cv_scores}')
print(f'Average CV MSE: {-cv_scores.mean()}')

# Save the model to disk if needed (Uncomment the following lines to use)
# import joblib
# joblib.dump(model, 'issues_closed_next_month_ridge_model.pkl')

# To use the model later, load it from the disk (Uncomment the following lines to use)
# loaded_model = joblib.load('issues_closed_next_month_ridge_model.pkl')
